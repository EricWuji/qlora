from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
import requests
import logging
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Support multiple API keys - can be set via environment variable or config file
ALLOWED_API_KEYS = set()

def load_api_keys():
    """Load API keys from environment variable or config"""
    # Method 1: From environment variable (comma-separated)
    env_keys = os.getenv("ALLOWED_API_KEYS", "")
    if env_keys:
        keys = [key.strip() for key in env_keys.split(",") if key.strip()]
        ALLOWED_API_KEYS.update(keys)
        logger.info(f"Loaded {len(keys)} API keys from environment")
    
    # Method 2: From file
    api_keys_file = os.getenv("API_KEYS_FILE", "./api_keys.txt")
    if os.path.exists(api_keys_file):
        try:
            with open(api_keys_file, 'r') as f:
                file_keys = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                ALLOWED_API_KEYS.update(file_keys)
                logger.info(f"Loaded {len(file_keys)} API keys from file")
        except Exception as e:
            logger.error(f"Error reading API keys file: {e}")
    
    # Fallback: Default API key
    if not ALLOWED_API_KEYS:
        default_key = os.getenv("API_KEY", "your-secret-api-key")
        ALLOWED_API_KEYS.add(default_key)
        logger.warning("No API keys configured, using default key")

# Load API keys on startup
load_api_keys()

def verify_api_key(api_key: str):
    """Verify API key from request body"""
    if api_key not in ALLOWED_API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")
    logger.info(f"Valid API key used: {api_key[:8]}...")
    return api_key

class ChatRequest(BaseModel):
    message: str
    api_key: str
    max_token: int = 100
    temperature: float = 0.7

@app.post("/chat")
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute
def chat_completion(
    request: Request,
    chat_request: ChatRequest
):
    # Verify API key from request body
    verify_api_key(chat_request.api_key)
    
    logger.info(f"Chat request received: message_length={len(chat_request.message)}, max_token={chat_request.max_token}, temperature={chat_request.temperature}")
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen2.5-7B-Instruct-merged",  # Use just the model name, not the path
                "prompt": f"### Instruction:\n{chat_request.message}\n### Response:\n",
                "max_tokens": chat_request.max_token,
                "temperature": chat_request.temperature,
                "echo": False
            },
            timeout=30  # 30 seconds timeout
        )
        response.raise_for_status()
        response_data = response.json()
        
        logger.info("Successfully received response from model")
        reply = response_data["choices"][0]["text"]
        logger.info(f"Reply generated: length={len(reply)}")
        
        return {"reply": reply}
        
    except requests.exceptions.Timeout:
        logger.error("Request timeout to model server")
        raise HTTPException(status_code=504, detail="Model server timeout")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(status_code=502, detail="Model server error")
    except KeyError as e:
        logger.error(f"Response format error: {str(e)}")
        raise HTTPException(status_code=502, detail="Invalid response format from model")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    return {"status": "healthy"}