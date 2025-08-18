import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Assistant: {response}")
    print("-" * 50)
