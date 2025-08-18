import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./models/Qwen2.5-7B-Instruct"
lora_path = "./out/Qwen2.5-7B-Instruct"
output_path = "./out/Qwen2.5-7B-Instruct-merged"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load and merge LoRA model
print("Loading LoRA model...")
model = PeftModel.from_pretrained(base_model, lora_path)

print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

# Save merged model
print("Saving merged model...")
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"Model successfully merged and saved to {output_path}")
