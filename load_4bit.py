from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_path = "./models/TinyLlama-1.1B-Chat-v1.0"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             quantization_config=quant_config).to(device)
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
print(f"已分配显存: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
print(model)