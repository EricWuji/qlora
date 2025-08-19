# LLM API Service

This project provides a FastAPI service with vLLM backend for serving LLM models.

## Environment Setup

### Using Conda (Recommended)

1. Create and activate conda environment:
```bash
conda create -n qlora python=3.10
conda activate qlora
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install vllm
```

3. Make sure your model is in the `out/` directory:
```bash
ls out/Qwen2.5-7B-Instruct-merged/
```

### Using System Python

1. Ensure Python 3.10+ is installed:
```bash
python3.10 --version
```

2. Install dependencies:
```bash
python3.10 -m pip install -r requirements.txt
python3.10 -m pip install vllm
```

## Quick Start

### Using Startup Script (Recommended)

1. Make the startup script executable:
```bash
chmod +x startup.sh
```

2. Start both vLLM and FastAPI services:
```bash
./startup.sh start
```

3. Check service status:
```bash
./startup.sh status
```

4. Test the API:
```bash
curl -X POST "http://localhost:8001/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Hello, how are you?",
       "api_key": "sk-1234567890abcdef1234567890abcdef",
       "max_token": 100,
       "temperature": 0.7
     }'
```

5. Stop services:
```bash
./startup.sh stop
```

### Manual Startup

1. Start vLLM server (in one terminal):
```bash
cd out
python -m vllm.entrypoints.openai.api_server \
    --model Qwen2.5-7B-Instruct-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```

2. Start FastAPI server (in another terminal):
```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

### Using Docker Compose

1. Build and start services:
```bash
docker-compose up --build
```

2. Test the API (same curl command as above)

## Startup Script Commands

```bash
./startup.sh start    # Start both services
./startup.sh stop     # Stop both services
./startup.sh restart  # Restart both services
./startup.sh status   # Show service status
./startup.sh logs vllm     # View vLLM logs
./startup.sh logs fastapi  # View FastAPI logs
```

## API Endpoints

- **POST /chat** - Chat completion endpoint
  - Request body: `{"message": "...", "api_key": "...", "max_token": 100, "temperature": 0.7}`
  - Response: `{"reply": "..."}`
- **GET /health** - Health check endpoint
  - Response: `{"status": "healthy"}`

## Configuration

### API Keys

Configure API keys in one of these ways:

1. **File-based** (edit `api_keys.txt`):
```
sk-1234567890abcdef1234567890abcdef
sk-abcdef1234567890abcdef1234567890
user-api-key-12345
```

2. **Environment variable**:
```bash
export ALLOWED_API_KEYS="key1,key2,key3"
```

3. **Custom file path**:
```bash
export API_KEYS_FILE="/path/to/your/keys.txt"
```

### Directory Structure

```
/home/wuyinqi/llm/qlora/
├── app.py              # FastAPI application
├── startup.sh          # Startup script
├── requirements.txt    # Python dependencies
├── api_keys.txt       # API keys configuration
├── logs/              # Log files
│   ├── vllm.log
│   └── fastapi.log
└── out/               # Model directory
    └── Qwen2.5-7B-Instruct-merged/
```

## Requirements

- Python 3.10+ or Conda environment
- NVIDIA GPU with CUDA support (for vLLM)
- Sufficient GPU memory for your model
- Model files in `out/` directory

## Troubleshooting

1. **Conda environment**: Make sure to activate your environment before running
2. **GPU memory**: Adjust `--gpu-memory-utilization` in startup.sh if needed
3. **Port conflicts**: Change ports in startup.sh if 8000/8001 are in use
4. **Model path**: Ensure model files are in `out/Qwen2.5-7B-Instruct-merged/`
5. **Logs**: Check `logs/vllm.log` and `logs/fastapi.log` for errors
