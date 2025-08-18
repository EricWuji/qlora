import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_PROJECT"] = "qlora-finetuning"
# os.environ["TORCH_LOAD_SKIP_SECURITY_CHECK"] = "1"  # Skip PyTorch security check for local files
# Add environment variable to suppress Flash Attention warnings if desired
# os.environ["FLASH_ATTENTION_SKIP_CUDA_CHECK"] = "1"

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
model_name = "./models/Qwen2.5-7B-Instruct"
dataset_name = "yahma/alpaca-cleaned"
output_dir = "./out/Qwen2.5-7B-Instruct2"
dataset_cache_dir = "./datasets/alpaca-cleaned"
logs_dir = "./logs"

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Change to bfloat16 to match training
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
    local_files_only=True,
    attn_implementation="flash_attention_2",
    # Remove use_safetensors=True to allow loading pytorch_model.bin files
    # Remove torch_dtype when using quantization to avoid conflicts
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
train_dataset = shuffled_dataset.select(range(100))  # First 1000 samples for training
eval_dataset = shuffled_dataset.select(range(100, 110))  # Next 100 samples for evaluation

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
    num_train_epochs=3,  # Increase epochs for better training
    per_device_train_batch_size=4,  # Reduce for memory efficiency with QLoRA
    per_device_eval_batch_size=4,  # Match train batch size
    gradient_accumulation_steps=8,  # Reduce to balance with smaller batch size
    warmup_steps=5,  # Keep proportional to max_steps
    max_steps=75,  # Increase for better convergence
    learning_rate=1e-4,  # Reduce learning rate for stable QLoRA training
    bf16=True,
    logging_steps=10,  # More frequent logging for monitoring
    save_steps=50,  # More frequent saves
    eval_steps=25,  # More frequent evaluation
    eval_strategy="steps",
    do_eval=True,
    optim="paged_adamw_32bit",  # Better optimizer for QLoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,  # Add warmup ratio
    weight_decay=0.01,  # Add weight decay for regularization
    max_grad_norm=1.0,  # Add gradient clipping
    remove_unused_columns=False,
    logging_dir=logs_dir,
    logging_strategy="steps",
    gradient_checkpointing=True,
    report_to="none",
    dataset_text_field="text",
    max_length=512,  # Correct parameter name for sequence length
    packing=False,
    dataloader_pin_memory=False,
    save_safetensors=True,  # Use safetensors format
    load_best_model_at_end=True,  # Load best model after training
    metric_for_best_model="eval_loss",  # Metric to determine best model
    greater_is_better=False,  # Lower loss is better
    save_total_limit=3,  # Limit number of saved checkpoints
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_eval_dataset,
    args=sft_config,
    processing_class=tokenizer,
    # Add data collator configuration for consistent dtypes
    data_collator=None,  # Let SFTTrainer handle it automatically
)

# Start training
print("Starting training...")
trainer.train()

# Save the model
trainer.save_model()
tokenizer.save_pretrained(output_dir)

print(f"Training completed! Model saved to {output_dir}")