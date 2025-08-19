#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VLLM_PORT=8000
FASTAPI_PORT=8001
MODEL_PATH="./out/Qwen2.5-7B-Instruct-merged"
MODEL_NAME="Qwen2.5-7B-Instruct-merged"  # Just the model name for vLLM
LOG_DIR="./logs"
VLLM_PID_FILE="$LOG_DIR/vllm.pid"
FASTAPI_PID_FILE="$LOG_DIR/fastapi.pid"

# Functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_python() {
    # Check if we're in a conda environment
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        print_success "Using conda environment: $CONDA_DEFAULT_ENV"
        PYTHON_CMD="python"
    elif command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        # Check if it's Python 3.10+
        python_version=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ $(echo "$python_version 3.10" | tr " " "\n" | sort -V | head -n1) != "3.10" ]]; then
            print_error "Python version is $python_version, but 3.10+ is required"
            exit 1
        fi
    else
        print_error "Python is not installed or not in PATH"
        print_error "Please activate your conda environment or install Python 3.10+"
        exit 1
    fi
    print_success "Python found: $($PYTHON_CMD --version)"
}

check_requirements() {
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_status "Installing/updating requirements..."
    $PYTHON_CMD -m pip install -r requirements.txt
    
    # Install vLLM separately
    print_status "Installing vLLM..."
    $PYTHON_CMD -m pip install vllm
}

check_model() {
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "Model not found at $MODEL_PATH"
        print_error "Please ensure your model is in the ./out/ directory"
        exit 1
    fi
    print_success "Model found at $MODEL_PATH"
}

create_log_dir() {
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        print_status "Created log directory: $LOG_DIR"
    fi
}

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Port $port is already in use"
        return 1
    fi
    return 0
}

start_vllm() {
    print_status "Starting vLLM server on port $VLLM_PORT..."
    
    # Change to out directory and start vLLM with relative model path
    cd ./out && nohup $PYTHON_CMD -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --gpu-memory-utilization 0.8 \
        --max-model-len 4096 \
        --served-model-name "$MODEL_NAME" \
        > "../$LOG_DIR/vllm.log" 2>&1 &
    
    # Return to original directory
    cd ..
    
    echo $! > "$VLLM_PID_FILE"
    print_success "vLLM server started with PID $(cat $VLLM_PID_FILE)"
    
    # Wait for vLLM to be ready
    print_status "Waiting for vLLM server to be ready..."
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
            print_success "vLLM server is ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "vLLM server failed to start within timeout"
    return 1
}

start_fastapi() {
    print_status "Starting FastAPI server on port $FASTAPI_PORT..."
    
    nohup $PYTHON_CMD -m uvicorn app:app \
        --host 0.0.0.0 \
        --port $FASTAPI_PORT \
        > "$LOG_DIR/fastapi.log" 2>&1 &
    
    echo $! > "$FASTAPI_PID_FILE"
    print_success "FastAPI server started with PID $(cat $FASTAPI_PID_FILE)"
    
    # Wait for FastAPI to be ready
    print_status "Waiting for FastAPI server to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$FASTAPI_PORT/health > /dev/null 2>&1; then
            print_success "FastAPI server is ready!"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    
    print_error "FastAPI server failed to start within timeout"
    return 1
}

stop_services() {
    print_status "Stopping services..."
    
    if [ -f "$VLLM_PID_FILE" ]; then
        local vllm_pid=$(cat "$VLLM_PID_FILE")
        if kill -0 $vllm_pid > /dev/null 2>&1; then
            kill $vllm_pid
            print_success "Stopped vLLM server (PID: $vllm_pid)"
        fi
        rm -f "$VLLM_PID_FILE"
    fi
    
    if [ -f "$FASTAPI_PID_FILE" ]; then
        local fastapi_pid=$(cat "$FASTAPI_PID_FILE")
        if kill -0 $fastapi_pid > /dev/null 2>&1; then
            kill $fastapi_pid
            print_success "Stopped FastAPI server (PID: $fastapi_pid)"
        fi
        rm -f "$FASTAPI_PID_FILE"
    fi
}

show_status() {
    echo -e "\n${BLUE}=== Service Status ===${NC}"
    
    if [ -f "$VLLM_PID_FILE" ]; then
        local vllm_pid=$(cat "$VLLM_PID_FILE")
        if kill -0 $vllm_pid > /dev/null 2>&1; then
            print_success "vLLM server is running (PID: $vllm_pid) on port $VLLM_PORT"
        else
            print_error "vLLM server is not running"
        fi
    else
        print_error "vLLM server is not running"
    fi
    
    if [ -f "$FASTAPI_PID_FILE" ]; then
        local fastapi_pid=$(cat "$FASTAPI_PID_FILE")
        if kill -0 $fastapi_pid > /dev/null 2>&1; then
            print_success "FastAPI server is running (PID: $fastapi_pid) on port $FASTAPI_PORT"
        else
            print_error "FastAPI server is not running"
        fi
    else
        print_error "FastAPI server is not running"
    fi
    
    echo -e "\n${BLUE}=== API Endpoints ===${NC}"
    echo -e "FastAPI: http://localhost:$FASTAPI_PORT"
    echo -e "Health Check: http://localhost:$FASTAPI_PORT/health"
    echo -e "vLLM OpenAI API: http://localhost:$VLLM_PORT"
    echo -e "\n${BLUE}=== Logs ===${NC}"
    echo -e "vLLM logs: $LOG_DIR/vllm.log"
    echo -e "FastAPI logs: $LOG_DIR/fastapi.log"
}

# Main script
case "$1" in
    start)
        print_status "Starting LLM API services..."
        
        check_python
        check_requirements
        check_model
        create_log_dir
        
        if ! check_port $VLLM_PORT; then
            print_error "Cannot start vLLM on port $VLLM_PORT"
            exit 1
        fi
        
        if ! check_port $FASTAPI_PORT; then
            print_error "Cannot start FastAPI on port $FASTAPI_PORT"
            exit 1
        fi
        
        if start_vllm && start_fastapi; then
            show_status
            print_success "All services started successfully!"
            echo -e "\n${YELLOW}To test the API, run:${NC}"
            echo -e "curl -X POST \"http://localhost:$FASTAPI_PORT/chat\" \\"
            echo -e "     -H \"Content-Type: application/json\" \\"
            echo -e "     -d '{"
            echo -e "       \"message\": \"Hello, how are you?\","
            echo -e "       \"api_key\": \"sk-1234567890abcdef1234567890abcdef\","
            echo -e "       \"max_token\": 100,"
            echo -e "       \"temperature\": 0.7"
            echo -e "     }'"
        else
            stop_services
            print_error "Failed to start services"
            exit 1
        fi
        ;;
    stop)
        stop_services
        print_success "All services stopped"
        ;;
    restart)
        stop_services
        sleep 2
        $0 start
        ;;
    status)
        show_status
        ;;
    logs)
        if [ "$2" = "vllm" ]; then
            tail -f "$LOG_DIR/vllm.log"
        elif [ "$2" = "fastapi" ]; then
            tail -f "$LOG_DIR/fastapi.log"
        else
            echo -e "${YELLOW}Usage: $0 logs [vllm|fastapi]${NC}"
            echo -e "Available logs:"
            echo -e "  vllm    - View vLLM server logs"
            echo -e "  fastapi - View FastAPI server logs"
        fi
        ;;
    *)
        echo -e "${YELLOW}Usage: $0 {start|stop|restart|status|logs}${NC}"
        echo -e ""
        echo -e "Commands:"
        echo -e "  start   - Start both vLLM and FastAPI servers"
        echo -e "  stop    - Stop both servers"
        echo -e "  restart - Restart both servers"
        echo -e "  status  - Show server status and endpoints"
        echo -e "  logs    - View server logs (specify vllm or fastapi)"
        exit 1
        ;;
esac
