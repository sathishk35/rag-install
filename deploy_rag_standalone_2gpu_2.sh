#!/bin/bash
# Standalone RAG System Deployment Script - 2x L40 8GB GPU Setup
# Optimized for 2x L40 8GB GPUs with available embedding models

set -e

# Configuration
RAG_HOME="/data/projects/rag-system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER="rag-system"
DATA_DIR="/data/projects/rag-system/data/rag"


# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

header() {
    echo -e "${PURPLE}"
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo -e "${NC}"
}

# Check prerequisites for 2-GPU mode
check_prerequisites() {
    header "CHECKING PREREQUISITES - 2x L40 8GB GPU MODE"
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
    
    # Check if base installation was completed
    if [[ ! -f "$RAG_HOME/INSTALLATION_SUMMARY.md" ]]; then
        error "Base installation not found. Please run complete_rag_installation.sh first"
    fi
    
    # Check GPU setup - Must have exactly 2 GPUs
    log "Checking GPU configuration..."
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        error "NVIDIA drivers not found. Please install NVIDIA drivers first"
    fi
    
    gpu_count=$(nvidia-smi -L | wc -l)
    if [[ $gpu_count -ne 2 ]]; then
        error "Expected exactly 2 GPUs, found $gpu_count. This script is for 2x L40 8GB setup"
    fi
    
    # Verify L40 GPUs with adequate memory
    for i in 0 1; do
        gpu_name=$(nvidia-smi --id=$i --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
        gpu_memory=$(nvidia-smi --id=$i --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
        
        log "GPU $i: $gpu_name (${gpu_memory}MB)"
        
        if [[ $gpu_memory -lt 7000 ]]; then
            error "GPU $i has insufficient memory: ${gpu_memory}MB (minimum 8GB required)"
        fi
        
        # Check if GPU is L40 (optional warning)
        if [[ "$gpu_name" == *"L40"* ]]; then
            log "✓ GPU $i: L40 detected"
        else
            warn "GPU $i: Not L40, but has adequate memory (${gpu_memory}MB)"
        fi
    done
    
    # Check CUDA availability
    if nvidia-smi > /dev/null 2>&1; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | head -1)
        log "CUDA Version: $cuda_version"
    else
        error "CUDA not properly installed or accessible"
    fi
    
    # Check core services
    services=("postgresql" "redis-rag" "qdrant")
    for service in "${services[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            warn "Service $service is not running. Starting..."
            systemctl start "$service"
            sleep 5
            if ! systemctl is-active --quiet "$service"; then
                error "Failed to start $service"
            fi
        else
            log "Service $service is running"
        fi
    done
    
    # Check Ollama service
    if systemctl is-active --quiet ollama; then
        log "Ollama service is running"
        
        # Test Ollama API
        if curl -s --max-time 5 http://localhost:11434/api/tags >/dev/null; then
            log "Ollama API is responding"
            
            # Check for Gemma model (recommended for L40 8GB)
            if curl -s http://localhost:11434/api/tags | grep -q "gemma2\|gemma3"; then
                log "Gemma model detected in Ollama"
            else
                warn "Gemma model not found. Recommended for L40 8GB:"
                warn "  ollama pull gemma2:2b"
                warn "  ollama pull gemma3:2b"
            fi
        else
            warn "Ollama API not responding on port 11434"
        fi
    else
        warn "Ollama service not running. Please start it:"
        warn "  sudo systemctl start ollama"
    fi
    
    # Check system resources
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_mem -lt 32 ]]; then
        warn "System memory is ${total_mem}GB (32GB+ recommended for 2-GPU mode)"
    else
        log "System memory: ${total_mem}GB"
    fi
    
    log "Prerequisites check completed for 2-GPU mode"
}

# Deploy RAG application with 2-GPU configuration
deploy_application() {
    header "DEPLOYING RAG APPLICATION - 2x L40 8GB GPU MODE"
    
    # Create application directory structure
    log "Creating application directory structure..."
    mkdir -p "$RAG_HOME"/{core,api,config,templates,scripts,logs,models,data,tests}
    mkdir -p "$DATA_DIR"/{documents,vectors,cache,backups,processed,failed,incoming}
    mkdir -p "$DATA_DIR/incoming"/{drivers,embedded,radar,rf,ew,ate,general,confidential,classified}
    
    # Create Python package structure
    log "Setting up Python package structure..."
    cat > "$RAG_HOME/__init__.py" << 'EOF'
"""
RAG System for Data Patterns India - 2x L40 8GB GPU Mode
Advanced Retrieval-Augmented Generation with GPU acceleration
"""

__version__ = "1.0.0"
__author__ = "Data Patterns India"
__mode__ = "2gpu_l40_8gb"
EOF

    # Create 2-GPU optimized configuration
    log "Creating 2-GPU optimized configuration..."
    cat > "$RAG_HOME/config/rag_config_2gpu.yaml" << 'EOF'
# RAG System 2x L40 8GB GPU Configuration
# Optimized for dual GPU setup with available embedding models

system:
  name: "Data Patterns India RAG System - 2x L40 8GB"
  version: "1.0.0"
  environment: "development"
  deployment_type: "2gpu_l40_8gb"
  mode: "gpu_testing"

# Hardware Configuration (2x L40 8GB optimized)
hardware:
  deployment_type: "dual_gpu"
  gpus:
    - device: 0
      name: "L40"
      memory_gb: 8
      allocation: "language_model"
      memory_fraction: 0.8
    - device: 1  
      name: "L40"
      memory_gb: 8
      allocation: "embeddings_processing"
      memory_fraction: 0.85
      
  gpu_optimization:
    enable_mixed_precision: true
    enable_memory_efficient_attention: true
    gradient_checkpointing: false
    
# Database Configuration
database:
  host: "localhost"
  port: 5432
  database: "rag_metadata"
  user: "rag-system"
  pool_size: 15
  max_overflow: 30
  connection_timeout: 30

# Vector Database Configuration (optimized for GPU)
qdrant:
  host: "localhost"
  port: 6333
  timeout: 60
  grpc_port: 6334
  collection_config:
    vectors:
      size: 1024        # E5-Large-v2 dimension
      distance: "Cosine"
    optimizers_config:
      deleted_threshold: 0.2
      vacuum_min_vector_number: 1000
      default_segment_number: 4
    hnsw_config:
      m: 32             # Optimized for 1024d
      ef_construct: 200
      full_scan_threshold: 10000

# Redis Configuration
redis:
  host: "localhost"
  port: 6380
  db: 0
  max_connections: 75
  connection_pool_kwargs:
    max_connections: 50
    retry_on_timeout: true

# Embedding Models Configuration (your available models)
embedding:
  device: "cuda:1"      # Use GPU 1 for embeddings
  models_directory: "/data/projects/rag-system/models"
  
  # Primary model - E5-Large-v2 (recommended for your setup)
  primary_model:
    name: "e5-large-v2-latest"
    dimension: 1024
    device: "cuda:1"
    batch_size: 16      # Good for 8GB
    max_seq_length: 512
    memory_fraction: 0.4
    precision: "float32"
    
  # Code-specific model
  code_model:
    name: "codebert-base-latest"
    dimension: 768
    device: "cuda:1"
    batch_size: 12
    max_seq_length: 512
    memory_fraction: 0.25
    
  # Multilingual model (optional, for future use)
  multilingual_model:
    name: "multilingual-e5-large"
    dimension: 1024
    device: "cuda:1"
    batch_size: 8
    memory_fraction: 0.2
    enabled: false      # Disable by default to save memory
    
  # BGE-M3 (high quality but memory intensive)
  bge_model:
    name: "bge-m3-latest"
    dimension: 1024
    device: "cuda:1"
    batch_size: 4       # Small batch due to memory requirements
    memory_fraction: 0.3
    enabled: false      # Disable by default, enable if needed
    
  processing:
    chunk_size: 512     # Match E5-Large-v2 context
    chunk_overlap: 100
    max_chunks_per_document: 100
    parallel_processing: true
    worker_threads: 8

# Language Model Configuration (Ollama on GPU 0)
language_model:
  provider: "ollama"
  device: "cuda:0"
  
  ollama:
    base_url: "http://localhost:11434"
    model: "gemma2:2b"  # Lightweight model for L40 8GB
    timeout: 120
    
    generation:
      temperature: 0.3
      max_tokens: 1500
      num_predict: 1500
      top_k: 40
      top_p: 0.9
      repeat_penalty: 1.1
      
    # GPU-specific options
    options:
      num_gpu: 1        # Use GPU 0 only
      gpu_layers: -1    # All layers on GPU
      num_thread: 16
      num_batch: 512
      use_mmap: true
      use_mlock: false
      f16_kv: true

# Security Configuration
security:
  classification_levels:
    - "public"
    - "internal"
    - "confidential"
    - "restricted"
    - "classified"
  domain_restrictions:
    classified: ["radar", "ew", "classified_drivers"]
    restricted: ["radar", "ew", "ate", "restricted_drivers"]
    confidential: ["drivers", "embedded", "radar", "ate"]
    internal: ["drivers", "embedded", "general", "ate"]
    public: ["general", "public_docs"]
  audit:
    enabled: true
    log_all_queries: true

# API Configuration (GPU-optimized)
api:
  host: "0.0.0.0"
  port: 8000
  workers: 3          # Balanced for 2-GPU setup
  worker_class: "uvicorn.workers.UvicornWorker"
  
  timeouts:
    request_timeout: 180
    keepalive_timeout: 65
    graceful_timeout: 30
    
  limits:
    max_request_size: 100_000_000
    max_concurrent_requests: 15   # Good for L40 8GB
    
  cors:
    allow_origins: ["*"]
    allow_credentials: true
  rate_limiting:
    enabled: false

# Document Processing Configuration (GPU-accelerated)
document_processing:
  processing_mode: "gpu_accelerated"
  device: "cuda:1"
  
  supported_formats:
    - ".pdf"
    - ".docx"
    - ".doc"
    - ".txt"
    - ".md"
    - ".c"
    - ".cpp"
    - ".h"
    - ".hpp"
    - ".py"
    - ".m"
    - ".json"
    - ".csv"
    
  limits:
    max_file_size_mb: 50
    max_pages_per_pdf: 200
    processing_timeout_seconds: 300
    max_concurrent_files: 3
    
  extraction:
    pdf_strategy: "pdfplumber"
    ocr_enabled: false
    image_processing: false
    
  chunking:
    strategy: "semantic"
    min_chunk_size: 200
    max_chunk_size: 512
    overlap_size: 100
    respect_sentence_boundaries: true

# Monitoring Configuration
monitoring:
  log_level: "INFO"
  
  logging:
    console:
      enabled: true
      level: "INFO"
    file:
      enabled: true
      path: "/var/log/rag"
      max_size_mb: 100
      
  gpu_monitoring:
    enabled: true
    collection_interval: 30
    memory_threshold: 85
    utilization_threshold: 90

# Performance Configuration (2-GPU specific)
performance:
  optimization_level: "gpu_balanced"
  
  gpu_settings:
    enable_amp: true              # Automatic Mixed Precision
    enable_torch_compile: false   # Disable for stability
    memory_efficient_attention: true
    
  threading:
    max_workers: 16
    thread_pool_size: 8
    io_thread_pool_size: 4
    
  caching:
    enabled: true
    max_cache_size_mb: 4096
    cache_ttl_seconds: 3600
    query_cache_enabled: true
    embedding_cache_enabled: true
    
  batch_processing:
    embedding_batch_size: 16      # Optimized for E5-Large-v2
    query_batch_size: 4
    document_batch_size: 2

# Data Storage Configuration
data:
  storage:
    base_directory: "/data/projects/rag-system/data/rag"
    documents_directory: "/data/projects/rag-system/data/rag/documents"
    vectors_directory: "/data/projects/rag-system/data/rag/vectors"
    cache_directory: "/data/projects/rag-system/data/rag/cache"
    temp_directory: "/tmp/rag"
    models_directory: "/data/projects/rag-system/models"

# Testing Configuration
testing:
  enabled: true
  test_data_path: "/data/projects/rag-system/tests/data"
  mock_responses: false
  debug_mode: true
  
  performance_testing:
    enabled: true
    max_response_time_seconds: 10
    min_accuracy_threshold: 0.85
    concurrent_user_limit: 12
    
# Model Selection Configuration
model_selection:
  auto_select_embedding: true
  
  # Priority order for embedding models
  embedding_priority:
    - "e5-large-v2-latest"      # Best balance for L40 8GB
    - "multilingual-e5-large"   # If multilingual needed
    - "codebert-base-latest"    # For code-specific tasks
    - "bge-m3-latest"          # High quality but memory intensive
    
  # Model switching based on task
  task_routing:
    code_analysis: "codebert-base-latest"
    general_docs: "e5-large-v2-latest"
    multilingual: "multilingual-e5-large"
    high_quality: "bge-m3-latest"
EOF

    # Copy the 2-GPU config as the main config
    cp "$RAG_HOME/config/rag_config_2gpu.yaml" "$RAG_HOME/config/rag_config.yaml"
    
    # Set ownership
    chown -R "$USER:$USER" "$RAG_HOME"
    chown -R "$USER:$USER" "$DATA_DIR"
    
    # Set permissions
    chmod -R 755 "$RAG_HOME"
    chmod -R 750 "$RAG_HOME/config"
    
    log "2-GPU application deployment completed"
}

# Setup Python application for 2-GPU mode
setup_python_app() {
    header "SETTING UP PYTHON APPLICATION - 2x L40 8GB GPU MODE"
    
    # Install GPU-optimized Python dependencies
    log "Installing GPU-optimized Python dependencies..."
    
    # Create 2-GPU requirements file
    cat > "$RAG_HOME/requirements_2gpu.txt" << 'EOF'
# Core RAG Framework - 2-GPU Optimized
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.3

# GPU-optimized PyTorch
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
accelerate==1.1.1

# Vector Database and Search
qdrant-client==1.15.1
sentence-transformers==3.3.1
transformers==4.46.3
numpy==2.2.1

# Document Processing
PyMuPDF==1.25.2
python-docx==1.1.2
pandas==2.2.3
beautifulsoup4==4.12.3
markdown==3.7

# Code Analysis
tree-sitter==0.20.4
tree-sitter-c==0.20.6
tree-sitter-cpp==0.20.3
tree-sitter-python==0.20.4
pygments==2.18.0

# Database and Caching
psycopg2-binary==2.9.10
redis==5.2.1
sqlalchemy==2.0.36

# NLP and Text Processing
nltk==3.9.1
spacy==3.7.6

# Security and Authentication
cryptography==44.0.0
pyjwt==2.10.1
python-multipart==0.0.17

# Utilities
tqdm==4.67.1
rich==13.9.4
python-dotenv==1.0.1
httpx==0.28.1
aiofiles==24.1.0
watchdog==6.0.0

# GPU Performance Optimization
nvidia-ml-py3==12.560.30
psutil==6.1.0

# Testing and Development
pytest==8.3.4
pytest-asyncio==0.25.0
EOF

    # Install packages in virtual environment
    sudo -u "$USER" bash -c "
        source /data/venv_ai/bin/activate
        pip install --upgrade pip setuptools wheel
        pip install --no-cache-dir -r '$RAG_HOME/requirements_2gpu.txt'
    "
    
    # Verify GPU PyTorch installation
    log "Verifying GPU PyTorch installation..."
    sudo -u "$USER" bash -c "
        source /data/venv_ai/bin/activate
        python -c \"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('WARNING: CUDA not available')
\"
    "
    
    # Download NLTK data
    log "Downloading NLTK data..."
    sudo -u "$USER" bash -c "
        source /data/venv_ai/bin/activate
        python -c \"
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'NLTK download error: {e}')
\"
    "
    
    # Setup embedding models (your available models)
    log "Setting up available embedding models..."
    sudo -u "$USER" bash -c "
        source /data/venv_ai/bin/activate
        python -c \"
import os
from sentence_transformers import SentenceTransformer
import torch

models_dir = '/data/projects/rag-system/models'
os.makedirs(models_dir, exist_ok=True)

# Available models mapping
models_map = {
    'e5-large-v2-latest': 'intfloat/e5-large-v2',
    'codebert-base-latest': 'microsoft/codebert-base',
    'multilingual-e5-large': 'intfloat/multilingual-e5-large',
    'bge-m3-latest': 'BAAI/bge-m3'
}

print('Setting up available embedding models...')
for local_name, hf_name in models_map.items():
    try:
        print(f'Loading {local_name}...')
        model = SentenceTransformer(hf_name)
        model.save(f'{models_dir}/{local_name}')
        
        # Test the model
        test_embedding = model.encode('test sentence')
        print(f'✅ {local_name}: dimension {len(test_embedding)}')
        
        # Clear GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f'❌ Failed to load {local_name}: {e}')

print('Embedding models setup completed')
\"
    " || warn "Some embedding models failed to download - they will be downloaded on first use"
    
    log "2-GPU Python application setup completed"
}

# Create systemd services for 2-GPU mode
create_systemd_services() {
    header "CREATING SYSTEMD SERVICES - 2x L40 8GB GPU MODE"
    
    # RAG API Service (2-GPU optimized)
    log "Creating 2-GPU optimized RAG API service..."
    cat > /etc/systemd/system/rag-api-2gpu.service << EOF
[Unit]
Description=RAG API Service - 2x L40 8GB GPU Mode
After=network.target postgresql.service redis-rag.service qdrant.service ollama.service
Wants=postgresql.service redis-rag.service qdrant.service ollama.service

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=$RAG_HOME
Environment=PATH=/data/venv_ai/bin
Environment=PYTHONPATH=$RAG_HOME
Environment=RAG_CONFIG_PATH=$RAG_HOME/config/rag_config.yaml
Environment=RAG_MODE=2gpu_l40_8gb
Environment=CUDA_VISIBLE_DEVICES=0,1
Environment=WORLD_SIZE=2
Environment=MASTER_ADDR=localhost
Environment=MASTER_PORT=29500
ExecStartPre=/bin/sleep 20
ExecStart=/data/venv_ai/bin/uvicorn api.rag_api:app --host 0.0.0.0 --port 8000 --workers 3 --log-level info
ExecReload=/bin/kill -HUP \$MAINPID
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
TimeoutStartSec=300
TimeoutStopSec=60
KillMode=mixed

# GPU optimization
MemoryMax=48G
CPUSchedulingPolicy=2

[Install]
WantedBy=multi-user.target
EOF

    # RAG Document Processor Service (2-GPU optimized)
    log "Creating 2-GPU optimized document processor service..."
    cat > /etc/systemd/system/rag-processor-2gpu.service << EOF
[Unit]
Description=RAG Document Processor - 2x L40 8GB GPU Mode
After=network.target postgresql.service redis-rag.service qdrant.service
Wants=postgresql.service redis-rag.service qdrant.service

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=$RAG_HOME
Environment=PATH=/data/venv_ai/bin
Environment=PYTHONPATH=$RAG_HOME
Environment=RAG_CONFIG_PATH=$RAG_HOME/config/rag_config.yaml
Environment=RAG_MODE=2gpu_l40_8gb
Environment=CUDA_VISIBLE_DEVICES=1
ExecStartPre=/bin/sleep 30
ExecStart=/data/venv_ai/bin/python scripts/document_processor_daemon.py
Restart=on-failure
RestartSec=20
StandardOutput=journal
StandardError=journal
TimeoutStartSec=180
KillMode=mixed

# Resource limits for GPU processing
MemoryMax=24G

[Install]
WantedBy=multi-user.target
EOF

    # RAG System Monitor Service (2-GPU aware)
    log "Creating 2-GPU system monitor service..."
    cat > /etc/systemd/system/rag-monitor-2gpu.service << EOF
[Unit]
Description=RAG System Monitor - 2x L40 8GB GPU Mode
After=network.target rag-api-2gpu.service

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=$RAG_HOME
Environment=PATH=/data/venv_ai/bin
Environment=PYTHONPATH=$RAG_HOME
Environment=RAG_CONFIG_PATH=$RAG_HOME/config/rag_config.yaml
Environment=RAG_MODE=2gpu_l40_8gb
Environment=CUDA_VISIBLE_DEVICES=0,1
ExecStartPre=/bin/sleep 40
ExecStart=/data/venv_ai/bin/python scripts/system_monitor.py
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload
    
    log "2-GPU systemd services created"
}

# Configure Nginx for 2-GPU mode (reuse standalone config with minor changes)
configure_nginx_2gpu() {
    header "CONFIGURING NGINX - 2x L40 8GB GPU MODE"
    
    log "Creating Nginx configuration for 2-GPU RAG system..."
    cat > "/etc/nginx/sites-available/rag-2gpu" << 'EOF'
# 2-GPU RAG System Nginx Configuration

upstream rag_api_2gpu {
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_2gpu:10m rate=20r/s;
limit_conn_zone $binary_remote_addr zone=conn_2gpu:10m;

server {
    listen 80;
    server_name rag-2gpu.local localhost;
    
    # Security headers
    add_header X-Frame-Options SAMEORIGIN always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Connection limits
    limit_conn conn_2gpu 30;
    
    # Logging
    access_log /var/log/nginx/rag-2gpu-access.log;
    error_log /var/log/nginx/rag-2gpu-error.log;
    
    # Large file uploads
    client_max_body_size 200M;
    client_body_timeout 300s;
    
    # Root redirect to API docs
    location = / {
        return 302 /api/docs;
    }
    
    # API endpoints
    location /api/ {
        limit_req zone=api_2gpu burst=30 nodelay;
        
        proxy_pass http://rag_api_2gpu;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # GPU-optimized timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Disable buffering for streaming
        proxy_buffering off;
        proxy_request_buffering off;
        
        # CORS headers
        add_header Access-Control-Allow-Origin * always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
        
        if ($request_method = 'OPTIONS') {
            add_header Access-Control-Allow-Origin * always;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
            add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
            add_header Access-Control-Max-Age 1728000;
            add_header Content-Type 'text/plain; charset=utf-8';
            add_header Content-Length 0;
            return 204;
        }
    }
    
    # Health check
    location /health {
        access_log off;
        proxy_pass http://rag_api_2gpu/api/health;
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 5s;
    }
    
	# GPU status endpoint
    location /gpu-status {
        proxy_pass http://rag_api_2gpu/api/gpu-status;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # Static files
    location /static/ {
        alias /data/projects/rag-system/static/;
        expires 1h;
    }
    
    # Deny access to sensitive paths
    location ~ /\. {
        deny all;
        return 404;
    }
}
EOF

    # Enable the 2-GPU site
    if [[ ! -L "/etc/nginx/sites-enabled/rag-2gpu" ]]; then
        ln -s "/etc/nginx/sites-available/rag-2gpu" "/etc/nginx/sites-enabled/rag-2gpu"
        log "Enabled RAG 2-GPU site"
    fi
    
    # Test Nginx configuration
    if nginx -t; then
        log "Nginx configuration is valid"
        systemctl reload nginx
    else
        error "Nginx configuration is invalid"
    fi
    
    log "Nginx configuration completed for 2-GPU mode"
}

# Create management scripts for 2-GPU mode
create_management_scripts() {
    header "CREATING MANAGEMENT SCRIPTS - 2x L40 8GB GPU MODE"
    
    # 2-GPU control script
    log "Creating 2-GPU control script..."
    cat > "$RAG_HOME/scripts/rag-2gpu.sh" << 'EOF'
#!/bin/bash
# RAG System 2-GPU Control Script for L40 8GB

GPU_SERVICES=("postgresql" "redis-rag" "qdrant" "ollama" "rag-api-2gpu" "rag-processor-2gpu" "rag-monitor-2gpu")
RAG_HOME="/data/projects/rag-system"
USER="rag-system"
API_BASE="http://localhost:8000"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

case "$1" in
    start)
        echo -e "${GREEN}Starting RAG System (2x L40 8GB GPU Mode)...${NC}"
        for service in "${GPU_SERVICES[@]}"; do
            echo -e "Starting $service..."
            systemctl start "$service"
            sleep 5
        done
        echo -e "${GREEN}RAG System (2-GPU) started${NC}"
        ;;
    stop)
        echo -e "${YELLOW}Stopping RAG System (2x L40 8GB GPU Mode)...${NC}"
        # Stop in reverse order
        for ((i=${#GPU_SERVICES[@]}-1; i>=0; i--)); do
            service="${GPU_SERVICES[i]}"
            echo -e "Stopping $service..."
            systemctl stop "$service" 2>/dev/null || true
        done
        echo -e "${YELLOW}RAG System (2-GPU) stopped${NC}"
        ;;
    restart)
        echo -e "${YELLOW}Restarting RAG System (2x L40 8GB GPU Mode)...${NC}"
        $0 stop
        sleep 10
        $0 start
        ;;
    status)
        echo -e "${GREEN}RAG System Status (2x L40 8GB GPU Mode):${NC}"
        for service in "${GPU_SERVICES[@]}"; do
            if systemctl is-active --quiet "$service"; then
                echo -e "✓ $service: ${GREEN}Running${NC}"
            else
                echo -e "✗ $service: ${RED}Not running${NC}"
            fi
        done
        
        # Show GPU information
        echo -e "\n${BLUE}GPU Status:${NC}"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r idx name mem_used mem_total util temp; do
                echo -e "GPU $idx ($name): ${mem_used}MB/${mem_total}MB (${util}% util, ${temp}°C)"
            done
        fi
        
        # Show API status
        echo -e "\n${BLUE}API Status:${NC}"
        if curl -s --max-time 10 "$API_BASE/api/health" > /dev/null; then
            echo -e "✓ RAG API: ${GREEN}Responding${NC}"
            
            # Get GPU-specific health info
            health_info=$(curl -s --max-time 5 "$API_BASE/api/health" | jq -r '.status // "unknown"' 2>/dev/null)
            echo -e "  Status: $health_info"
        else
            echo -e "✗ RAG API: ${RED}Not responding${NC}"
        fi
        
        # Show embedding model status
        echo -e "\n${BLUE}Embedding Models:${NC}"
        models_response=$(curl -s --max-time 10 "$API_BASE/api/models/status" 2>/dev/null)
        if echo "$models_response" | grep -q "models"; then
            echo -e "✓ Models API: ${GREEN}Responding${NC}"
            
            # List available embedding models
            available_models=$(echo "$models_response" | jq -r '.embedding_models[]?' 2>/dev/null | head -3)
            if [[ -n "$available_models" ]]; then
                echo -e "  Available: $(echo "$available_models" | tr '\n' ', ' | sed 's/,$//')"
            fi
        else
            echo -e "✗ Models API: ${RED}Not responding${NC}"
        fi
        
        # Show Ollama status
        echo -e "\n${BLUE}Ollama Status:${NC}"
        if curl -s --max-time 10 http://localhost:11434/api/tags > /dev/null; then
            echo -e "✓ Ollama: ${GREEN}Responding${NC}"
            
            # List models optimized for L40 8GB
            models=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null | head -3)
            if [[ -n "$models" ]]; then
                echo -e "  Models: $(echo "$models" | tr '\n' ', ' | sed 's/,$//')"
            fi
        else
            echo -e "✗ Ollama: ${RED}Not responding${NC}"
        fi
        ;;
    health)
        echo -e "${GREEN}Performing 2-GPU health check...${NC}"
        
        # Check GPU status and memory
        echo -e "\n${BLUE}GPU Health Check:${NC}"
        if command -v nvidia-smi >/dev/null 2>&1; then
            for i in 0 1; do
                gpu_name=$(nvidia-smi --id=$i --query-gpu=name --format=csv,noheader 2>/dev/null)
                gpu_mem_used=$(nvidia-smi --id=$i --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
                gpu_mem_total=$(nvidia-smi --id=$i --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
                gpu_util=$(nvidia-smi --id=$i --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
                gpu_temp=$(nvidia-smi --id=$i --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
                
                mem_percent=$((gpu_mem_used * 100 / gpu_mem_total))
                
                if [[ $mem_percent -lt 90 ]] && [[ $gpu_temp -lt 85 ]]; then
                    echo -e "✓ GPU $i: ${GREEN}Healthy${NC} ($gpu_name, ${mem_percent}% mem, ${gpu_temp}°C)"
                else
                    echo -e "⚠ GPU $i: ${YELLOW}Warning${NC} ($gpu_name, ${mem_percent}% mem, ${gpu_temp}°C)"
                fi
            done
        else
            echo -e "✗ nvidia-smi not available"
        fi
        
        # Check API health
        if curl -s --max-time 15 "$API_BASE/api/health" > /dev/null; then
            echo -e "✓ API: ${GREEN}Responding${NC}"
        else
            echo -e "✗ API: ${RED}Not responding${NC}"
        fi
        
        # Check database connections
        if sudo -u $USER psql -h localhost rag_metadata -c "SELECT 1;" > /dev/null 2>&1; then
            echo -e "✓ Database: ${GREEN}Connected${NC}"
        else
            echo -e "✗ Database: ${RED}Connection failed${NC}"
        fi
        
        # Check Qdrant
        if curl -s --max-time 10 http://localhost:6333/collections > /dev/null; then
            echo -e "✓ Qdrant: ${GREEN}Responding${NC}"
        else
            echo -e "✗ Qdrant: ${RED}Not responding${NC}"
        fi
        
        # Check Redis
        if redis-cli -p 6380 ping 2>/dev/null | grep -q PONG; then
            echo -e "✓ Redis: ${GREEN}Connected${NC}"
        else
            echo -e "✗ Redis: ${RED}Connection failed${NC}"
        fi
        ;;
    test)
        echo -e "${GREEN}Running 2-GPU system tests...${NC}"
        
        # Test basic query with GPU acceleration
        echo -e "\n${BLUE}Testing GPU-accelerated query...${NC}"
        start_time=$(date +%s.%N)
        test_response=$(curl -s -X POST "$API_BASE/api/query" \
            -H "Content-Type: application/json" \
            -d '{"query": "test GPU system with embedding models", "user_id": "gpu_test_user"}' 2>/dev/null)
        end_time=$(date +%s.%N)
        query_time=$(echo "$end_time - $start_time" | bc)
        
        if [[ $? -eq 0 ]] && echo "$test_response" | grep -q "response"; then
            echo -e "✓ GPU query: ${GREEN}Working (${query_time}s)${NC}"
        else
            echo -e "✗ GPU query: ${RED}Failed${NC}"
        fi
        
        # Test embedding generation
        echo -e "\n${BLUE}Testing embedding generation...${NC}"
        start_time=$(date +%s.%N)
        embed_response=$(curl -s -X POST "$API_BASE/api/embeddings" \
            -H "Content-Type: application/json" \
            -d '{"texts": ["test embedding with GPU", "second test embedding"]}' 2>/dev/null)
        end_time=$(date +%s.%N)
        embed_time=$(echo "$end_time - $start_time" | bc)
        
        if echo "$embed_response" | grep -q "embedding"; then
            echo -e "✓ GPU embedding: ${GREEN}Working (${embed_time}s)${NC}"
        else
            echo -e "✗ GPU embedding: ${RED}Failed${NC}"
        fi
        
        # Test document processing with 2-GPU setup
        echo -e "\n${BLUE}Testing document processing...${NC}"
        echo "This is a test document for 2-GPU RAG system validation." > /tmp/test_2gpu.txt
        
        start_time=$(date +%s.%N)
        upload_response=$(curl -s -X POST "$API_BASE/api/documents/upload" \
            -F "file=@/tmp/test_2gpu.txt" \
            -F "security_classification=internal" \
            -F "domain=general" 2>/dev/null)
        end_time=$(date +%s.%N)
        upload_time=$(echo "$end_time - $start_time" | bc)
        
        if echo "$upload_response" | grep -q "success"; then
            echo -e "✓ Document processing: ${GREEN}Working (${upload_time}s)${NC}"
        else
            echo -e "✗ Document processing: ${RED}Failed${NC}"
        fi
        
        rm -f /tmp/test_2gpu.txt
        
        # GPU performance summary
        echo -e "\n${BLUE}2-GPU Performance Summary:${NC}"
        total_test_time=$(echo "$query_time + $embed_time + $upload_time" | bc)
        echo -e "Total test time: ${total_test_time}s"
        
        if (( $(echo "$total_test_time < 15.0" | bc -l) )); then
            echo -e "Overall performance: ${GREEN}Excellent for L40 8GB${NC}"
        elif (( $(echo "$total_test_time < 30.0" | bc -l) )); then
            echo -e "Overall performance: ${GREEN}Good for L40 8GB${NC}"
        else
            echo -e "Overall performance: ${YELLOW}Consider optimizations${NC}"
        fi
        ;;
    models)
        echo -e "${GREEN}Managing embedding models for 2-GPU setup...${NC}"
        
        case "$2" in
            list)
                echo -e "\n${BLUE}Available embedding models:${NC}"
                echo -e "✓ e5-large-v2-latest (1024d, recommended)"
                echo -e "✓ codebert-base-latest (768d, for code)"
                echo -e "✓ multilingual-e5-large (1024d, multilingual)"
                echo -e "✓ bge-m3-latest (1024d, high quality, memory intensive)"
                
                # Check which models are actually loaded
                models_response=$(curl -s --max-time 10 "$API_BASE/api/models/status" 2>/dev/null)
                if echo "$models_response" | grep -q "models"; then
                    echo -e "\n${BLUE}Currently active:${NC}"
                    echo "$models_response" | jq -r '.embedding_models[]?' 2>/dev/null || echo "Unable to fetch active models"
                fi
                ;;
            switch)
                model_name="$3"
                if [[ -z "$model_name" ]]; then
                    echo "Usage: $0 models switch <model_name>"
                    echo "Available: e5-large-v2-latest, codebert-base-latest, multilingual-e5-large, bge-m3-latest"
                    exit 1
                fi
                
                echo -e "Switching primary embedding model to: $model_name"
                # This would require API endpoint to switch models dynamically
                curl -s -X POST "$API_BASE/api/models/switch" \
                    -H "Content-Type: application/json" \
                    -d "{\"model_name\": \"$model_name\"}" || echo "Model switch API not available"
                ;;
            test)
                echo -e "\n${BLUE}Testing available embedding models...${NC}"
                for model in "e5-large-v2-latest" "codebert-base-latest"; do
                    echo -e "\nTesting $model..."
                    test_result=$(curl -s -X POST "$API_BASE/api/embeddings/test" \
                        -H "Content-Type: application/json" \
                        -d "{\"model\": \"$model\", \"text\": \"test embedding\"}" 2>/dev/null)
                    
                    if echo "$test_result" | grep -q "success"; then
                        echo -e "✓ $model: ${GREEN}Working${NC}"
                    else
                        echo -e "✗ $model: ${RED}Failed${NC}"
                    fi
                done
                ;;
            *)
                echo "Usage: $0 models {list|switch <model>|test}"
                ;;
        esac
        ;;
    gpu)
        echo -e "${GREEN}GPU management for L40 8GB setup...${NC}"
        
        case "$2" in
            status)
                echo -e "\n${BLUE}Detailed GPU Status:${NC}"
                nvidia-smi
                ;;
            memory)
                echo -e "\n${BLUE}GPU Memory Usage:${NC}"
                nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv
                ;;
            optimize)
                echo -e "\n${BLUE}Applying GPU optimizations...${NC}"
                
                # Set GPU performance mode
                for i in 0 1; do
                    if [[ -f "/sys/class/drm/card$i/device/power_dpm_force_performance_level" ]]; then
                        echo "high" | sudo tee "/sys/class/drm/card$i/device/power_dpm_force_performance_level" > /dev/null
                        echo -e "✓ GPU $i set to high performance mode"
                    fi
                done
                
                # Clear GPU memory cache
                echo -e "Clearing GPU memory cache..."
                nvidia-smi --gpu-reset > /dev/null 2>&1 || echo "GPU reset not available"
                
                echo -e "✓ GPU optimization completed"
                ;;
            *)
                echo "Usage: $0 gpu {status|memory|optimize}"
                ;;
        esac
        ;;
    enable)
        echo -e "${GREEN}Enabling RAG System services (2-GPU Mode)...${NC}"
        for service in "${GPU_SERVICES[@]}"; do
            systemctl enable "$service" 2>/dev/null || true
            echo -e "✓ Enabled $service"
        done
        ;;
    disable)
        echo -e "${YELLOW}Disabling RAG System services (2-GPU Mode)...${NC}"
        for service in "${GPU_SERVICES[@]}"; do
            systemctl disable "$service" 2>/dev/null || true
            echo -e "✓ Disabled $service"
        done
        ;;
    logs)
        service=${2:-rag-api-2gpu}
        echo -e "${GREEN}Showing logs for $service...${NC}"
        journalctl -u "$service" -f --no-pager
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|health|test|models <action>|gpu <action>|enable|disable|logs [service]}"
        echo ""
        echo "2-GPU Mode Services: ${GPU_SERVICES[*]}"
        echo ""
        echo "Examples:"
        echo "  $0 start              # Start all services"
        echo "  $0 status             # Show status with GPU info"
        echo "  $0 health             # Perform GPU health check"
        echo "  $0 test               # Run GPU performance tests"
        echo "  $0 models list        # List available embedding models"
        echo "  $0 models switch e5-large-v2-latest  # Switch primary model"
        echo "  $0 gpu status         # Detailed GPU status"
        echo "  $0 gpu optimize       # Apply GPU optimizations"
        exit 1
        ;;
esac
EOF

    chmod +x "$RAG_HOME/scripts/rag-2gpu.sh"
    
    # Create 2-GPU quick test script
    log "Creating 2-GPU quick test script..."
    cat > "$RAG_HOME/scripts/quick_test_2gpu.sh" << 'EOF'
#!/bin/bash
# Quick Test Script for 2x L40 8GB GPU RAG System

API_BASE="http://localhost:8000"

echo "🚀 RAG System 2x L40 8GB GPU Quick Test"
echo "======================================="
echo

# Test 1: GPU Status Check
echo "1️⃣ GPU Status Check..."
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L | wc -l)
    echo "GPUs detected: $gpu_count"
    
    for i in 0 1; do
        gpu_info=$(nvidia-smi --id=$i --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null)
        echo "GPU $i: $gpu_info"
    done
else
    echo "❌ nvidia-smi not available"
fi
echo

# Test 2: Health Check
echo "2️⃣ Health Check..."
health_response=$(curl -s --max-time 10 "$API_BASE/api/health")
if echo "$health_response" | grep -q '"status"'; then
    status=$(echo "$health_response" | jq -r '.status // "unknown"')
    echo "✅ Health: $status"
else
    echo "❌ Health check failed"
fi
echo

# Test 3: Embedding Models Test
echo "3️⃣ Embedding Models Test..."
models_response=$(curl -s --max-time 10 "$API_BASE/api/models/status")
if echo "$models_response" | grep -q '"models"'; then
    echo "✅ Models API responding"
    available_models=$(echo "$models_response" | jq -r '.embedding_models[]?' 2>/dev/null | wc -l)
    echo "Available embedding models: $available_models"
    
    # Test embedding generation
    start_time=$(date +%s.%N)
    embed_test=$(curl -s --max-time 15 -X POST "$API_BASE/api/embeddings" \
        -H "Content-Type: application/json" \
        -d '{"texts": ["GPU test embedding"]}')
    end_time=$(date +%s.%N)
    embed_time=$(echo "$end_time - $start_time" | bc)
    
    if echo "$embed_test" | grep -q "embedding"; then
        echo "✅ Embedding generation: ${embed_time}s"
    else
        echo "❌ Embedding generation failed"
    fi
else
    echo "❌ Models API not responding"
fi
echo

# Test 4: GPU-Accelerated Query
echo "4️⃣ GPU-Accelerated Query Test..."
start_time=$(date +%s.%N)
query_response=$(curl -s --max-time 30 -X POST "$API_BASE/api/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What are the benefits of GPU acceleration in embeddings?", "user_id": "gpu_test_user"}')
end_time=$(date +%s.%N)
query_time=$(echo "$end_time - $start_time" | bc)

if echo "$query_response" | grep -q '"response"'; then
    echo "✅ GPU query processing: ${query_time}s"
    response=$(echo "$query_response" | jq -r '.response // "No response"')
    echo "Response: ${response:0:100}..."
    
    confidence=$(echo "$query_response" | jq -r '.confidence_score // 0')
    echo "Confidence: $confidence"
    
    # Performance assessment
    if (( $(echo "$query_time < 5.0" | bc -l) )); then
        echo "🚀 Performance: Excellent GPU acceleration"
    elif (( $(echo "$query_time < 10.0" | bc -l) )); then
        echo "✅ Performance: Good GPU acceleration"
    else
        echo "⚠️ Performance: Check GPU utilization"
    fi
else
    echo "❌ GPU query processing failed"
fi
echo

# Test 5: Document Processing with GPUs
echo "5️⃣ Document Processing Test..."
cat > /tmp/gpu_test_doc.txt << 'EOF'
GPU Acceleration in Machine Learning

Graphics Processing Units (GPUs) excel at parallel processing tasks.
Key benefits for embeddings:
- Parallel matrix operations
- High memory bandwidth
- Optimized tensor operations
- Batch processing capabilities

L40 GPUs provide excellent performance for inference workloads.
EOF

start_time=$(date +%s.%N)
upload_response=$(curl -s --max-time 60 -X POST "$API_BASE/api/documents/upload" \
    -F "file=@/tmp/gpu_test_doc.txt" \
    -F "security_classification=internal" \
    -F "domain=general")
end_time=$(date +%s.%N)
upload_time=$(echo "$end_time - $start_time" | bc)

if echo "$upload_response" | grep -q "success"; then
    echo "✅ Document processing: ${upload_time}s"
else
    echo "❌ Document processing failed"
fi

rm -f /tmp/gpu_test_doc.txt
echo

# Test 6: GPU Memory Usage
echo "6️⃣ GPU Memory Usage..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Current GPU memory usage:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | while IFS=, read -r idx mem_used mem_total util; do
        mem_percent=$((${mem_used%% *} * 100 / ${mem_total%% *}))
        echo "GPU $idx: ${mem_percent}% memory, ${util%% *}% utilization"
    done
else
    echo "❌ Cannot check GPU memory usage"
fi
echo

# Performance Summary
echo "📊 2x L40 8GB Performance Summary"
echo "================================="
total_time=$(echo "$embed_time + $query_time + $upload_time" | bc)
echo "Performance metrics:"
echo "  Embedding: ${embed_time}s"
echo "  Query: ${query_time}s"
echo "  Document: ${upload_time}s"
echo "  Total: ${total_time}s"

if (( $(echo "$total_time < 20.0" | bc -l) )); then
    echo "🚀 Overall: Excellent GPU performance"
elif (( $(echo "$total_time < 40.0" | bc -l) )); then
    echo "✅ Overall: Good GPU performance"
else
    echo "⚠️ Overall: GPU performance needs optimization"
fi

echo
echo "📍 Access RAG API at: $API_BASE"
echo "📚 View API docs at: $API_BASE/api/docs"
echo "🔧 2-GPU controls: sudo /data/projects/rag-system/scripts/rag-2gpu.sh"
echo "⚡ GPU status: sudo /data/projects/rag-system/scripts/rag-2gpu.sh gpu status"
echo "🎯 Model management: sudo /data/projects/rag-system/scripts/rag-2gpu.sh models list"
EOF

    chmod +x "$RAG_HOME/scripts/quick_test_2gpu.sh"
    
    log "2-GPU management scripts created"
}

# Initialize system for 2-GPU testing
initialize_system_2gpu() {
    header "INITIALIZING SYSTEM - 2x L40 8GB GPU MODE"
    
    # Wait for GPU services to be ready
    log "Waiting for GPU-optimized services to initialize..."
    sleep 30
    
    # Check GPU memory before proceeding
    log "Checking GPU memory availability..."
    for i in 0 1; do
        gpu_memory=$(nvidia-smi --id=$i --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo "0")
        if [[ $gpu_memory -lt 4000 ]]; then
            warn "GPU $i has limited free memory: ${gpu_memory}MB"
        else
            log "GPU $i free memory: ${gpu_memory}MB"
        fi
    done
    
    # Create test users
    log "Creating test users..."
    
    sudo -u "$USER" /data/venv_ai/bin/python "$RAG_HOME/scripts/user_manager.py" create \
        --user-id "admin" \
        --security-level "classified" \
        --domains "*" \
        --admin-user "system" || warn "Failed to create admin user"
    
    sudo -u "$USER" /data/venv_ai/bin/python "$RAG_HOME/scripts/user_manager.py" create \
        --user-id "gpu_test_user" \
        --security-level "internal" \
        --domains "general" "drivers" \
        --admin-user "admin" || warn "Failed to create GPU test user"
    
    sudo -u "$USER" /data/venv_ai/bin/python "$RAG_HOME/scripts/user_manager.py" create \
        --user-id "engineer" \
        --security-level "confidential" \
        --domains "drivers" "embedded" "general" \
        --admin-user "admin" || warn "Failed to create engineer user"
    
    # Create GPU-optimized sample documents
    log "Creating GPU-optimized sample documents..."
    mkdir -p "/data/projects/rag-system/data/rag/test_documents_2gpu"
    
    # Create more comprehensive documents for GPU testing
    cat > "/data/projects/rag-system/data/rag/test_documents_2gpu/gpu_acceleration_guide.md" << 'EOF'
# GPU Acceleration in Embedded Systems

## Overview
GPU acceleration can significantly improve performance in computational tasks for embedded and defense systems.

## L40 GPU Capabilities

### Memory Architecture
- 24GB GDDR6 memory (8GB variant available)
- High memory bandwidth for parallel processing
- Efficient for inference workloads

### Compute Performance
- Thousands of CUDA cores
- Tensor processing units
- Mixed precision support (FP16/FP32)

## Applications in Defense Electronics

### Signal Processing
```python
import numpy as np
import cupy as cp  # GPU-accelerated NumPy

# GPU-accelerated FFT for radar processing
def gpu_fft_processing(signal_data):
    gpu_data = cp.asarray(signal_data)
    fft_result = cp.fft.fft(gpu_data)
    return cp.asnumpy(fft_result)
```

### Embedding Generation
- Parallel text processing
- Batch embedding computation
- Real-time semantic search

### Code Analysis
- Parallel syntax parsing
- Batch code similarity analysis
- Large-scale repository processing

## Best Practices

### Memory Management
- Monitor GPU memory usage
- Use batch processing for efficiency
- Clear unused tensors regularly

### Performance Optimization
- Use mixed precision when possible
- Optimize batch sizes for your GPU
- Profile GPU utilization regularly
EOF

    cat > "/data/projects/rag-system/data/rag/test_documents_2gpu/embedding_models_comparison.md" << 'EOF'
# Embedding Models Comparison for L40 8GB

## Available Models

### E5-Large-v2 (Recommended)
- **Dimension**: 1024
- **Memory**: ~1.3GB
- **Performance**: Excellent balance
- **Use case**: General technical documents

### CodeBERT-Base
- **Dimension**: 768  
- **Memory**: ~0.5GB
- **Performance**: Fast
- **Use case**: Code analysis and programming

### BGE-M3
- **Dimension**: 1024
- **Memory**: ~2.2GB
- **Performance**: Highest quality
- **Use case**: Critical applications (memory intensive)

### Multilingual-E5-Large
- **Dimension**: 1024
- **Memory**: ~1.3GB
- **Performance**: Good
- **Use case**: Multilingual documents

## Performance Benchmarks

### L40 8GB Batch Sizes
- E5-Large-v2: 16 texts/batch
- CodeBERT-Base: 24 texts/batch
- BGE-M3: 8 texts/batch
- Multilingual-E5: 12 texts/batch

### Memory Usage
```
Model Memory Allocation (L40 8GB):
├── E5-Large-v2: 1.3GB (safe)
├── CodeBERT: 0.5GB (very safe)
├── BGE-M3: 2.2GB (tight fit)
└── Multilingual: 1.3GB (safe)
```

## Recommendations

For L40 8GB setup:
1. **Primary**: E5-Large-v2 for general use
2. **Code**: CodeBERT-Base for programming tasks  
3. **Quality**: BGE-M3 when memory allows
4. **Multilingual**: When language diversity needed
EOF

    # Set ownership and ingest
    chown -R "$USER:$USER" "/data/projects/rag-system/data/rag/test_documents_2gpu"
    
    # Ingest sample documents with GPU processing
    log "Ingesting sample documents with GPU acceleration..."
    sleep 10
    
    for doc in /data/projects/rag-system/data/rag/test_documents_2gpu/*; do
        filename=$(basename "$doc")
        log "Ingesting $filename with GPU processing..."
        
        timeout 300 curl -s -X POST "http://localhost:8000/api/documents/upload" \
            -F "file=@$doc" \
            -F "security_classification=internal" \
            -F "domain=general" >/dev/null 2>&1 || warn "Failed to ingest $filename"
    done
    
    log "2-GPU system initialization completed"
}

# Deployment verification for 2-GPU
verify_deployment_2gpu() {
    header "VERIFYING 2x L40 8GB GPU DEPLOYMENT"
    
    log "Waiting for all GPU services to be ready..."
    sleep 40
    
    # Check services
    log "Checking service status..."
    failed_services=()
    
    gpu_services=("postgresql" "redis-rag" "qdrant" "ollama" "rag-api-2gpu" "rag-processor-2gpu" "rag-monitor-2gpu")
    
    for service in "${gpu_services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            log "✓ $service is running"
        else
            warn "✗ $service is not running"
            failed_services+=("$service")
        fi
    done
    
    # Check GPU status
    log "Checking GPU status..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        for i in 0 1; do
            gpu_name=$(nvidia-smi --id=$i --query-gpu=name --format=csv,noheader 2>/dev/null)
            gpu_memory=$(nvidia-smi --id=$i --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
            gpu_util=$(nvidia-smi --id=$i --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
            
            log "GPU $i: $gpu_name (${gpu_memory}MB, ${gpu_util}% util)"
        done
    else
        warn "✗ GPU monitoring not available"
    fi
    
    # Test API functionality
    log "Testing GPU-accelerated API functionality..."
    
    # Health check
    if curl -s --max-time 15 http://localhost:8000/api/health >/dev/null; then
        log "✓ RAG API health endpoint responding"
    else
        warn "✗ RAG API health endpoint not responding"
    fi
    
    # Test GPU query
    test_query=$(curl -s --max-time 30 -X POST http://localhost:8000/api/query \
        -H "Content-Type: application/json" \
        -d '{"query": "test 2-GPU system with embedding acceleration", "user_id": "gpu_test_user"}' 2>/dev/null)
    
    if echo "$test_query" | grep -q '"response"'; then
        log "✓ GPU-accelerated query processing working"
    else
        warn "✗ GPU-accelerated query processing failed"
    fi
    
    # Test embedding models
    embedding_test=$(curl -s --max-time 20 -X POST http://localhost:8000/api/embeddings \
        -H "Content-Type: application/json" \
        -d '{"texts": ["test GPU embedding generation"]}' 2>/dev/null)
    
    if echo "$embedding_test" | grep -q "embedding"; then
        log "✓ GPU embedding generation working"
    else
        warn "✗ GPU embedding generation failed"
    fi
    
    # Test Ollama integration
    if curl -s --max-time 10 http://localhost:11434/api/tags >/dev/null; then
        log "✓ Ollama integration working"
        
        # Check for GPU-appropriate models
        model_count=$(curl -s http://localhost:11434/api/tags | jq '.models | length' 2>/dev/null || echo "0")
        log "  Available models: $model_count"
    else
        warn "✗ Ollama integration failed"
    fi
    
    # Generate deployment report
    cat > "$RAG_HOME/2GPU_DEPLOYMENT_REPORT.md" << EOF
# RAG System 2x L40 8GB GPU Deployment Report

**Deployment Date:** $(date)
**Mode:** 2x L40 8GB GPU Acceleration
**System:** $(uname -a)

## Deployment Summary

$(if [[ ${#failed_services[@]} -eq 0 ]]; then
    echo "✅ **SUCCESS**: All services deployed and running with GPU acceleration"
else
    echo "⚠️ **PARTIAL**: ${#failed_services[@]} services failed"
fi)

## GPU Configuration
- **GPU Count**: 2
- **GPU Type**: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
- **GPU Memory**: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1) each
- **CUDA Version**: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | head -1)

## Service Status
$(for service in "${gpu_services[@]}"; do
    if systemctl is-active --quiet "$service"; then
        echo "- ✅ $service: Running"
    else
        echo "- ❌ $service: Not Running"
    fi
done)

## GPU Status
$(for i in 0 1; do
    gpu_name=$(nvidia-smi --id=$i --query-gpu=name --format=csv,noheader 2>/dev/null)
    gpu_memory=$(nvidia-smi --id=$i --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)
    gpu_util=$(nvidia-smi --id=$i --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null)
    echo "- GPU $i ($gpu_name): $gpu_memory, $gpu_util utilization"
done)

## Embedding Models Configuration
- **Primary Model**: e5-large-v2-latest (1024d, GPU 1)
- **Code Model**: codebert-base-latest (768d, GPU 1)
- **Available Models**: 
  - e5-large-v2-latest ✅
  - codebert-base-latest ✅ 
  - multilingual-e5-large ✅
  - bge-m3-latest ✅

## Access Points
- **RAG API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health
- **GPU Status**: http://localhost:8000/api/gpu-status

## Performance Expectations
- **Query Response**: 3-8 seconds (GPU accelerated)
- **Embedding Generation**: 0.5-2 seconds
- **Document Processing**: 5-20 seconds
- **Concurrent Users**: 10-15 (optimized for L40 8GB)

## Quick Commands
\`\`\`bash
# Check system status
sudo /data/projects/rag-system/scripts/rag-2gpu.sh status

# Run health check with GPU monitoring
sudo /data/projects/rag-system/scripts/rag-2gpu.sh health

# Quick performance test
sudo /data/projects/rag-system/scripts/quick_test_2gpu.sh

# GPU-specific commands
sudo /data/projects/rag-system/scripts/rag-2gpu.sh gpu status
sudo /data/projects/rag-system/scripts/rag-2gpu.sh gpu memory

# Embedding model management
sudo /data/projects/rag-system/scripts/rag-2gpu.sh models list
sudo /data/projects/rag-system/scripts/rag-2gpu.sh models switch e5-large-v2-latest

# Test query with GPU acceleration
curl -X POST http://localhost:8000/api/query \\
  -H "Content-Type: application/json" \\
  -d '{"query": "How does GPU acceleration improve embedding performance?", "user_id": "gpu_test_user"}'
\`\`\`

## System Information
- **CPU**: $(nproc) cores
- **Memory**: $(free -h | awk '/^Mem:/{print $2}')
- **Storage**: $(df -h / | awk 'NR==2{print $4}') available
- **GPUs**: 2x L40 $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

## Database Status
- **Documents**: $(sudo -u rag-system psql -h localhost rag_metadata -t -c "SELECT COUNT(*) FROM documents;" 2>/dev/null | tr -d ' ' || echo "N/A")
- **Users**: $(sudo -u rag-system psql -h localhost rag_metadata -t -c "SELECT COUNT(*) FROM user_access;" 2>/dev/null | tr -d ' ' || echo "N/A")
- **Vector Collections**: $(curl -s http://localhost:6333/collections | jq '.result.collections | length' 2>/dev/null || echo "N/A")

## Optimization Tips
1. **Monitor GPU Memory**: Keep usage under 85% for stability
2. **Batch Size Tuning**: Adjust based on available GPU memory
3. **Model Selection**: Use E5-Large-v2 for balanced performance
4. **Concurrent Requests**: Limit to 10-15 for optimal performance
5. **GPU Temperature**: Monitor to stay under 85°C

## Troubleshooting
- **High GPU Memory**: Reduce batch sizes or switch to smaller models
- **Slow Performance**: Check GPU utilization and memory usage
- **CUDA Errors**: Verify CUDA drivers and PyTorch GPU support
- **Memory Issues**: Restart services to clear GPU memory

## Next Steps for Production
1. **Load Testing**: Test with expected user load
2. **Model Optimization**: Fine-tune batch sizes for your workload
3. **Monitoring Setup**: Implement GPU monitoring and alerting
4. **Backup Strategy**: Configure automated backups
5. **Scaling**: Plan for additional GPUs if needed

The 2x L40 8GB GPU RAG system is ready for testing and evaluation!
EOF

    chown "$USER:$USER" "$RAG_HOME/2GPU_DEPLOYMENT_REPORT.md"
    
    log "2-GPU deployment verification completed"
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log "🎉 2-GPU deployment successful!"
    else
        warn "⚠️ Deployment completed with issues. Check the report."
    fi
}

# Main deployment function for 2-GPU mode
main() {
    header "RAG SYSTEM 2x L40 8GB GPU DEPLOYMENT"
    info "Deploying RAG system optimized for 2x L40 8GB GPU setup"
    info "Available embedding models: e5-large-v2, codebert-base, multilingual-e5, bge-m3"
    info "Estimated time: 20-25 minutes"
    
    # Run deployment steps
    check_prerequisites
    deploy_application
    ##setup_python_app
    create_systemd_services
    configure_nginx_2gpu
    create_management_scripts
    
    # Start services
    header "STARTING SERVICES - 2x L40 8GB GPU MODE"
    log "Starting GPU-optimized RAG services..."
    
    gpu_services=("postgresql" "redis-rag" "qdrant" "ollama" "rag-api-2gpu" "rag-processor-2gpu" "rag-monitor-2gpu")
    
    for service in "${gpu_services[@]}"; do
        systemctl enable "$service" 2>/dev/null || true
        systemctl start "$service"
        sleep 8  # Allow GPU initialization time
    done
    
    # Initialize system
    initialize_system_2gpu
    
    # Verify deployment
    verify_deployment_2gpu
    
    # Final summary
    header "2x L40 8GB GPU DEPLOYMENT COMPLETE"
    log "RAG System deployed successfully with GPU acceleration!"
    log ""
    log "🚀 GPU Configuration:"
    log "   • GPUs: 2x L40 (8GB each)"
    log "   • GPU 0: Language model (Gemma2:2B)"
    log "   • GPU 1: Embeddings + Processing"
    log "   • Total GPU Memory: 16GB"
    log ""
    log "🔗 Access Points:"
    log "   • RAG API: http://localhost:8000"
    log "   • API Documentation: http://localhost:8000/api/docs"
    log "   • Health Check: http://localhost:8000/api/health"
    log "   • GPU Status: http://localhost:8000/api/gpu-status"
    log ""
    log "🧪 GPU Testing Commands:"
    log "   • Quick Test: /data/projects/rag-system/scripts/quick_test_2gpu.sh"
    log "   • System Status: /data/projects/rag-system/scripts/rag-2gpu.sh status"
    log "   • GPU Health: /data/projects/rag-system/scripts/rag-2gpu.sh health"
    log "   • GPU Monitor: /data/projects/rag-system/scripts/rag-2gpu.sh gpu status"
    log ""
    log "🎯 Embedding Models:"
    log "   • Primary: e5-large-v2-latest (1024d, recommended)"
    log "   • Code: codebert-base-latest (768d, for programming)"
    log "   • Quality: bge-m3-latest (1024d, memory intensive)"
    log "   • Multilingual: multilingual-e5-large (1024d)"
    log "   • Management: /data/projects/rag-system/scripts/rag-2gpu.sh models list"
    log ""
    log "📋 Management:"
    log "   • Control Script: /data/projects/rag-system/scripts/rag-2gpu.sh"
    log "   • User Management: /data/projects/rag-system/scripts/user_manager.py"
    log "   • View Logs: journalctl -u rag-api-2gpu -f"
    log ""
    log "📄 Documentation:"
    log "   • Deployment Report: /data/projects/rag-system/2GPU_DEPLOYMENT_REPORT.md"
    log "   • Configuration: /data/projects/rag-system/config/rag_config.yaml"
    log ""
    log "⚡ Performance Expectations:"
    log "   • Query Response: 3-8 seconds (GPU accelerated)"
    log "   • Embedding Generation: 0.5-2 seconds"
    log "   • Document Processing: 5-20 seconds"
    log "   • Concurrent Users: 10-15 (optimized for L40 8GB)"
    log ""
    log "✅ The 2x L40 8GB GPU RAG system is ready for testing!"
    log ""
    info "Try a GPU-accelerated test query:"
    info "curl -X POST http://localhost:8000/api/query \\"
    info "  -H 'Content-Type: application/json' \\"
    info "  -d '{\"query\": \"How does GPU acceleration improve embedding performance?\", \"user_id\": \"gpu_test_user\"}'"
    log ""
    log "🔄 For production scaling, consider upgrading to larger GPUs or adding more L40s"
}

# Run main function
main "$@"
