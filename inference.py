import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Path to the merged model
model_path = "./out/Qwen2.5-7B-Instruct-merged"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading merged model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Create streamer for streaming output
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("Model loaded successfully!")
print("Type 'quit' or 'exit' to stop the conversation.")
print("-" * 50)

while True:
    user_input = input("User: ")
    
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break
    
    if not user_input.strip():
        continue
    
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    
    print("Assistant: ", end="", flush=True)
    
    # Generate response with streaming
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )
    
    print("\n" + "-" * 50)
