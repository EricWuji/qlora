import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Model and dataset configuration
model_name = "./models/TinyLlama-1.1B-Chat-v1.0"
dataset_name = "yahma/alpaca-cleaned"
output_dir = "./output"
dataset_cache_dir = "./datasets/alpaca-cleaned"
logs_dir = "./logs"

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False,
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare model for training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Always apply PEFT configuration for quantized models
print("Applying PEFT configuration to quantized model...")
model = get_peft_model(model, lora_config)
print("PEFT configuration applied successfully")

# Load and preprocess dataset
dataset = load_dataset(dataset_name, cache_dir=dataset_cache_dir)

# Shuffle the dataset and split into train/eval
shuffled_dataset = dataset["train"].shuffle(seed=42)
train_dataset = shuffled_dataset.select(range(1000))  # First 1000 samples for training
eval_dataset = shuffled_dataset.select(range(1000, 1100))  # Next 100 samples for evaluation

def format_instruction(sample):
    if sample["input"]:
        return f"""### Instruction:
{sample["instruction"]}

### Input:
{sample["input"]}

### Response:
{sample["output"]}"""
    else:
        return f"""### Instruction:
{sample["instruction"]}

### Response:
{sample["output"]}"""

# Format the dataset
def format_dataset(sample):
    sample["text"] = format_instruction(sample)
    return sample

formatted_train_dataset = train_dataset.map(format_dataset, remove_columns=train_dataset.column_names, num_proc=4)
formatted_eval_dataset = eval_dataset.map(format_dataset, remove_columns=eval_dataset.column_names, num_proc=4)

# Training arguments using SFTConfig
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Reduce batch size
    gradient_accumulation_steps=16,  # Increase to maintain effective batch size
    warmup_steps=100,
    max_steps=1000,
    learning_rate=2e-4,
    bf16=True,  # Use bf16 instead of fp16
    logging_steps=25,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    do_eval=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    logging_dir=logs_dir,
    logging_strategy="steps",
    gradient_checkpointing=True,
    report_to="none",  # Disable wandb/tensorboard
    dataset_text_field="text", # Field in the dataset containing the text
    max_length=512,
    packing=False,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_eval_dataset,
    args=sft_config,
    processing_class=tokenizer,
)

# Start training
print("Starting training...")
trainer.train()

# Save the model
trainer.save_model()
tokenizer.save_pretrained(output_dir)

print(f"Training completed! Model saved to {output_dir}")