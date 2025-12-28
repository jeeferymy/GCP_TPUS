import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    set_seed,
)
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# --- 1. Configuration ---
# Use a wrapper function for the main training logic. This is required for xmp.spawn.
# CORRECTED: The function must accept an 'index' argument from xmp.spawn.
def train_t5_on_tpu(index):
    """
    Main function to set up and run the training process on a TPU core.
    The 'index' argument is passed by xmp.spawn and represents the core ID.
    """
    # Set seed for reproducibility
    set_seed(42)

    # Model and dataset names
    MODEL_CHECKPOINT = "mesolitica/nanot5-base-malaysian-cased"
    DATASET_NAME = "mesolitica/Malaysian-Translation"
    
    # Your GCS bucket for storing results
    # Replace with your actual bucket name
    GCS_BUCKET = "gs://ejen-sayang-training-bucket-01" 
    OUTPUT_DIR = os.path.join(GCS_BUCKET, "nanot5-malay-translation-finetuned-v5litepod")

    # Training hyperparameters
    # On a 1-core TPU, you have more memory per core. You can likely increase this.
    BATCH_SIZE = 32  # Per TPU core.
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    LOGGING_STEPS = 50
    SAVE_STEPS = 500 # Save more frequently on a single core
    SOURCE_LANG = "src"
    TARGET_LANG = "tgt"

    # --- 2. Load Model and Tokenizer ---
    # The model is loaded on the correct TPU device by the Trainer automatically.
    # We just need to load the model and tokenizer objects.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # --- 3. Load and Preprocess Dataset ---
    print(f"[Core {index}] Loading dataset...")
    raw_datasets = load_dataset(DATASET_NAME,"stage1")

    # T5 models often benefit from adding a task-specific prefix.
    # For translation, a common format is "translate Source to Target: ..."
    prefix = f"translate {SOURCE_LANG} to {TARGET_LANG}: "

    def preprocess_function(examples):
        """Tokenize the source and target texts."""
        inputs = [prefix + ex for ex in examples["en"]]
        targets = examples["ms"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print(f"[Core {index}] Preprocessing dataset...")
    # Apply the preprocessing to the entire dataset
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    
    # Select a smaller subset for quick demonstration. Remove this for full training.
    # tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

    # --- 4. Set up the Trainer ---
    # The DataCollator will dynamically pad the sentences to the longest length in a batch.
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Get the number of TPU cores (this will be 1 for v5litepod-1)
    tpu_cores = xm.xrt_world_size()
    print(f"[Core {index}] Running on {tpu_cores} TPU core(s).")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        
        # TPU specific arguments
        tpu_num_cores=tpu_cores,

        # Training hyperparameters
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        
        # Logging and saving
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        
        # Evaluation
        # evaluation_strategy="epoch", # Uncomment if you have a validation set
        
        # Report to none to avoid extra dependencies. Can be "tensorboard", "wandb" etc.
        report_to="none", 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        # eval_dataset=tokenized_datasets["validation"], # Uncomment if you have a validation set
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 5. Start Training ---
    print(f"[Core {index}] Starting training...")
    trainer.train()
    
    # --- 6. Save the Final Model ---
    # The trainer.save_model() handles saving on the master process (index 0).
    print(f"[Core {index}] Saving final model to GCS...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[Core {index}] Training complete!")


# --- 7. Main Execution Block ---
# This is the entry point for the script.
if __name__ == "__main__":
    # We use nprocs=1 because the v5litepod-1 has only 1 TPU core.
    # xmp.spawn launches the training function on each available TPU core,
    # passing the core index as the first argument.
    xmp.spawn(train_t5_on_tpu, args=(), nprocs=1)
