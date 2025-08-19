# LLM API Service

This project provides a FastAPI service with vLLM backend for serving LLM models.

## Quick Start

### Using Docker Compose (Recommended)

1. Build and start services:
```bash
docker-compose up --build
```

2. Test the API:
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

### Using Docker

1. Build image:
```bash
docker build -t llm-api .
```

2. Run container:
```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 llm-api
```

## API Endpoints

- **POST /chat** - Chat completion endpoint
- **GET /health** - Health check endpoint

## Configuration

- Models should be placed in `out/` directory
- API keys can be configured via environment variables or `api_keys.txt`
- Logs are written to `logs/` directory

## Requirements

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
