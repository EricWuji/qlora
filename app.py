from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    max_token: int = 100
    temperature: float = 0.7

@app.post("/chat")
def chat_completion(request: ChatRequest):
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "Qwen2.5-7B-Instruct-merged",
            "prompt": f"### Instruction:\n{request.message}\n### Response:\n",
            "max_tokens": request.max_token,
            "temperature": request.temperature,
            "echo": False
        }
    ).json()
    print(reply)
    print("-" * 50)
    reply = response["choice"][0]["text"]
    return {"reply": reply}