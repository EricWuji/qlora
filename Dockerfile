FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Install vLLM
RUN python3.10 -m pip install vllm

# Copy application files
COPY app.py .
COPY api_keys.txt .

# Copy models directory (assuming models are in out/)
COPY out/ ./models/

# Create logs directory
RUN mkdir -p logs

# Create startup script
RUN cat > start.sh << 'EOF'
#!/bin/bash

# Start vLLM server in background
echo "Starting vLLM server..."
python3.10 -m vllm.entrypoints.openai.api_server \
    --model ./models/Qwen2.5-7B-Instruct-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --served-model-name Qwen2.5-7B-Instruct-merged &

# Wait for vLLM to start
echo "Waiting for vLLM server to start..."
sleep 30

# Check if vLLM is running
until curl -f http://localhost:8000/health > /dev/null 2>&1; do
    echo "Waiting for vLLM server..."
    sleep 5
done

echo "vLLM server is ready!"

# Start FastAPI application
echo "Starting FastAPI application..."
python3.10 -m uvicorn app:app --host 0.0.0.0 --port 8001
EOF

RUN chmod +x start.sh

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start both services
CMD ["./start.sh"]
