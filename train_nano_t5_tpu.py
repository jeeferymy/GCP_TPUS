# train_nano_t5_tpu.py
import os
import functools
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
from flax import traverse_util
from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration
from datasets import load_dataset

from nano_t5_config import NanoT5Config

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "nano-t5"
OUTPUT_DIR = "gs://your-bucket/nano-t5-checkpoints"  # Must be GCS bucket
MAX_SEQ_LEN = 128
BATCH_SIZE_PER_DEVICE = 16
NUM_TRAIN_EPOCHS = 3

# ----------------------------
# TPU Setup
# ----------------------------
try:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
except:
    pass  # Running on Cloud TPU VM

num_devices = jax.device_count()
global_batch_size = BATCH_SIZE_PER_DEVICE * num_devices
print(f"Global batch size: {global_batch_size} across {num_devices} devices")

# ----------------------------
# Data Loading
# ----------------------------
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["target"]
    model_inputs = tokenizer(
        inputs, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True
    )
    labels = tokenizer(
        targets, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True
    ).input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Example: load a simple dataset (e.g., translation or summarization)
dataset = load_dataset("xsum", split="train[:10000]")  # Tiny slice for demo
dataset = dataset.rename_columns({"document": "input", "summary": "target"})
dataset = dataset.map(preprocess_function, batched=True)
dataset.set_format(type="jax", columns=["input_ids", "attention_mask", "labels"])

# ----------------------------
# Model & State
# ----------------------------
config = NanoT5Config()
model = FlaxT5ForConditionalGeneration(config=config, dtype=jnp.bfloat16)

def create_train_state(rng, model, learning_rate=1e-3):
    params = model.params
    tx = optax.adamw(learning_rate=learning_rate)
    return train_state.TrainState.create(apply_fn=model.__call__, params=params, tx=tx)

# ----------------------------
# Training Step
# ----------------------------
@functools.partial(jax.pmap, axis_name="batch")
def train_step(state, batch):
    def loss_fn(params):
        labels = batch["labels"]
        logits = state.apply_fn(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=params,
            train=True
        ).logits
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    state = state.apply_gradients(grads=grads)
    metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
    return state, metrics

# ----------------------------
# Main Training Loop
# ----------------------------
def main():
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model)
    state = flax.jax_utils.replicate(state)

    dataloader = dataset.iter(batch_size=global_batch_size)
    steps_per_epoch = len(dataset) // global_batch_size

    for epoch in range(NUM_TRAIN_EPOCHS):
        for step, batch in enumerate(dataloader):
            batch = jax.tree_map(lambda x: x.reshape(num_devices, -1, *x.shape[1:]), batch)
            state, metrics = train_step(state, batch)
            if step % 50 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {metrics['loss'][0]:.4f}")

        # Optional: save checkpoint (to GCS)
        if jax.process_index() == 0:
            params = flax.jax_utils.unreplicate(state.params)
            model.save_pretrained(save_directory=OUTPUT_DIR, params=params)
            tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
