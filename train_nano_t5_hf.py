# train_nano_t5_hf.py
import os
import functools
from typing import Dict, Any

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
from transformers import (
    T5Tokenizer,
    FlaxT5ForConditionalGeneration,
    DataCollatorForT5MLM,
    AutoTokenizer,
)
from datasets import load_dataset

from nano_t5_config import NanoT5Config

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "nano-t5"
OUTPUT_DIR = "gs://your-bucket/nano-t5-c4"  # Replace with your GCS bucket
DATASET_NAME = "c4"  # or "wikitext", "bookcorpus", etc.
DATASET_CONFIG = "en"  # for c4; use "wikitext-103-raw-v1" for wikitext
MAX_SEQ_LEN = 512
BATCH_SIZE_PER_DEVICE = 8  # Adjust based on TPU memory
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 1e-3

# ----------------------------
# TPU Setup
# ----------------------------
try:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
except:
    pass

num_devices = jax.device_count()
global_batch_size = BATCH_SIZE_PER_DEVICE * num_devices
print(f"Global batch size: {global_batch_size} on {num_devices} TPU cores")

# ----------------------------
# Load Tokenizer & Dataset
# ----------------------------
tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=MAX_SEQ_LEN)

# Load dataset
if DATASET_NAME == "wikitext":
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:50000]")
    text_column = "text"
elif DATASET_NAME == "c4":
    dataset = load_dataset("c4", "en", split="train[:50000]", streaming=False)
    text_column = "text"
else:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train[:50000]")
    text_column = "text"  # adjust if needed

# Filter empty lines
dataset = dataset.filter(lambda x: len(x[text_column].strip()) > 0)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples[text_column],
        return_special_tokens_mask=True,
        add_special_tokens=True,
        truncation=True,
        padding=False,  # We'll pad dynamically in collator
        max_length=MAX_SEQ_LEN,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing",
)

# ----------------------------
# T5 Span Corruption (MLM-style)
# ----------------------------
# Hugging Face doesn't expose DataCollatorForT5MLM directly, so we define it
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class DataCollatorForT5MLM:
    tokenizer: Any
    noise_density: float = 0.15
    mean_noise_span_length: float = 3.0
    input_length: int = 512
    sentinel_token_ids: List[int] = None

    def __post_init__(self):
        if self.sentinel_token_ids is None:
            self.sentinel_token_ids = list(
                range(self.tokenizer.vocab_size, self.tokenizer.vocab_size + 100)
            )

    def __call__(self, examples):
        # Pad to max length
        batch = self.tokenizer.pad(
            examples, padding=True, max_length=self.input_length, return_tensors="np"
        )
        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = self.random_spans_noise_mask(expandend_input_length)
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        inputs = self.filter_input_ids(input_ids, input_ids_sentinel)
        labels = self.filter_input_ids(input_ids, labels_sentinel)

        # Pad inputs and labels to same length
        max_input_len = min(self.input_length, inputs.shape[-1])
        max_label_len = min(self.input_length, labels.shape[-1])

        inputs = inputs[:, :max_input_len]
        labels = labels[:, :max_label_len]

        # Pad to fixed length
        def pad_to_len(x, L):
            if x.shape[-1] < L:
                x = np.pad(x, ((0,0), (0, L - x.shape[-1])), constant_values=self.tokenizer.pad_token_id)
            return x[:, :L]

        inputs = pad_to_len(inputs, self.input_length)
        labels = pad_to_len(labels, self.input_length)

        return {"input_ids": inputs, "labels": labels}

    def random_spans_noise_mask(self, length):
        """Create span corruption mask (T5 style)"""
        num_noise_tokens = int(np.round(length * self.noise_density))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)

        # Place markers for start of spans
        random_start = np.random.randint(1, length - num_noise_spans)
        span_starts = np.random.choice(
            np.arange(1, length - num_noise_spans + 1),
            size=num_noise_spans,
            replace=False
        )
        span_starts = np.sort(span_starts)

        # Determine span lengths
        span_lengths = np.full(num_noise_spans, self.mean_noise_span_length, dtype=int)
        # Adjust last span to not exceed total noise tokens
        diff = num_noise_tokens - span_lengths.sum()
        if diff != 0:
            span_lengths[-1] += diff

        # Create mask
        noise_mask = np.zeros(length, dtype=bool)
        for start, span_len in zip(span_starts, span_lengths):
            noise_mask[start : start + span_len] = True

        return noise_mask

    def create_sentinel_ids(self, mask_indices):
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), mask_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, len(self.tokenizer) - sentinel_ids, 0
        )
        sentinel_ids -= mask_indices - 1
        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        batch_size = input_ids.shape[0]
        input_ids_full = []
        for idx in range(batch_size):
            input_id = input_ids[idx]
            sentinel_id = sentinel_ids[idx]
            mask = sentinel_id != 0
            input_id = input_id[mask]
            sentinel_id = sentinel_id[mask]
            input_id = np.where(sentinel_id < 0, input_id, sentinel_id)
            input_ids_full.append(input_id)
        # Pad to max length
        max_len = max(len(x) for x in input_ids_full)
        padded = np.full((batch_size, max_len), self.tokenizer.pad_token_id)
        for i, x in enumerate(input_ids_full):
            padded[i, : len(x)] = x
        return padded


# Create collator
collator = DataCollatorForT5MLM(
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_noise_span_length=3.0,
    input_length=MAX_SEQ_LEN,
)

# Convert dataset to batched NumPy for JAX
def collate_fn(examples):
    return collator(examples)

# Use a simple batcher (since we're on TPU VM)
def get_dataloader(dataset, batch_size):
    def gen():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            yield collate_fn([dict(zip(batch.keys(), row)) for row in zip(*batch.values())])
    return gen

# ----------------------------
# Model & Train State
# ----------------------------
config = NanoT5Config()
model = FlaxT5ForConditionalGeneration(config=config, dtype=jnp.bfloat16)

def create_train_state(rng, model, learning_rate):
    params = model.params
    tx = optax.adamw(learning_rate=learning_rate)
    return train_state.TrainState.create(apply_fn=model.__call__, params=params, tx=tx)

# ----------------------------
# Training Step
# ----------------------------
@functools.partial(jax.pmap, axis_name="batch")
def train_step(state, batch):
    labels = batch["labels"]

    def loss_fn(params):
        logits = state.apply_fn(
            input_ids=batch["input_ids"],
            decoder_input_ids=labels,
            params=params,
            train=True
        ).logits
        # Shift for T5: labels are already correct
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        )
        # Ignore pad tokens
        mask = (labels != tokenizer.pad_token_id)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name="batch")
    return state, loss

# ----------------------------
# Main
# ----------------------------
def main():
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, model, LEARNING_RATE)
    state = flax.jax_utils.replicate(state)

    dataloader = get_dataloader(tokenized_dataset, global_batch_size)
    total_steps = len(tokenized_dataset) // global_batch_size

    for epoch in range(NUM_TRAIN_EPOCHS):
        for step, batch in enumerate(dataloader()):
            # Move to devices
            batch = jax.tree_map(
                lambda x: x.reshape(num_devices, -1, *x.shape[1:]),
                batch
            )
            state, loss = train_step(state, batch)
            if step % 20 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss[0]:.4f}")

        # Save checkpoint
        if jax.process_index() == 0:
            params = flax.jax_utils.unreplicate(state.params)
            model.save_pretrained(save_directory=OUTPUT_DIR, params=params)
            tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
