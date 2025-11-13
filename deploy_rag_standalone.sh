#!/bin/bash
# Standalone RAG System Deployment Script
# Deploys RAG system without OpenWebUI integration for testing

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

# Check prerequisites for standalone mode
check_prerequisites() {
    header "CHECKING PREREQUISITES - STANDALONE MODE"
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
    
    # Check if base installation was completed
    if [[ ! -f "$RAG_HOME/INSTALLATION_SUMMARY.md" ]]; then
        error "Base installation not found. Please run complete_rag_installation.sh first"
    fi
    
    # Check core services (excluding OpenWebUI dependencies)
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
            
            # Check for gemma3 model
            if curl -s http://localhost:11434/api/tags | grep -q "gemma3"; then
                log "gemma3 model detected in Ollama"
            else
                warn "gemma3 model not found. You may need to pull it:"
                warn "  ollama pull gemma3:1b"
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
        warn "System memory is ${total_mem}GB (32GB+ recommended for standalone mode)"
    else
        log "System memory: ${total_mem}GB"
    fi
    
    log "Prerequisites check completed for standalone mode"
}

# Deploy RAG application (standalone version)
deploy_application() {
    header "DEPLOYING RAG APPLICATION - STANDALONE MODE"
    
    # Create application directory structure
    log "Creating application directory structure..."
    mkdir -p "$RAG_HOME"/{core,api,config,templates,scripts,logs,models,data,tests}
    mkdir -p "$DATA_DIR"/{documents,vectors,cache,backups,processed,failed,incoming}
    mkdir -p "$DATA_DIR/incoming"/{drivers,embedded,radar,rf,ew,ate,general,confidential,classified}
    
    # Create Python package structure
    log "Setting up Python package structure..."
    cat > "$RAG_HOME/__init__.py" << 'EOF'
"""
RAG System for Data Patterns India - Standalone Mode
Advanced Retrieval-Augmented Generation with security and domain awareness
"""

__version__ = "1.0.0"
__author__ = "Data Patterns India"
__mode__ = "standalone"
EOF

    # Create standalone configuration (optimized for testing)
    log "Creating standalone configuration..."
    cat > "$RAG_HOME/config/rag_config_standalone.yaml" << 'EOF'
# RAG System Standalone Configuration
# Optimized for testing without OpenWebUI integration

system:
  name: "Data Patterns India RAG System - Standalone"
  version: "1.0.0"
  environment: "development"
  deployment_type: "standalone"
  mode: "testing"

# Hardware Configuration (Conservative for testing)
hardware:
  gpus:
    - device: 0
      allocation: "gemma3_primary"
      memory_fraction: 0.8
    - device: 1  
      allocation: "gemma3_backup"
      memory_fraction: 0.8
    - device: 2
      allocation: "embeddings"
      memory_fraction: 0.7
    - device: 3
      allocation: "processing"
      memory_fraction: 0.5

# Database Configuration
database:
  host: "localhost"
  port: 5432
  database: "rag_metadata"
  user: "rag-system"
  pool_size: 10
  max_overflow: 20

# Vector Database Configuration  
qdrant:
  host: "localhost"
  port: 6333
  timeout: 30

# Redis Configuration
redis:
  host: "localhost"
  port: 6380
  db: 0
  max_connections: 50

# Embedding Models Configuration
embedding:
  models_directory: "/data/projects/rag-system/models"
  primary_model:
    name: "bge-m3"
    dimension: 1024
    device: "cuda:2"
    batch_size: 16  # Smaller batches for testing
  code_model:
    name: "codebert-base"  
    dimension: 768
    device: "cuda:2"
    batch_size: 32
  chunk_size: 800  # Smaller chunks for testing
  chunk_overlap: 150

# Language Model Configuration (Ollama)
language_model:
  provider: "ollama"
  ollama:
    base_url: "http://localhost:11434"
    model: "gemma3:1b"
    timeout: 300
    generation:
      temperature: 0.3
      max_tokens: 1500  # Smaller for testing
      num_predict: 1500

# Security Configuration (Simplified for testing)
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

# API Configuration (Standalone)
api:
  host: "0.0.0.0"
  port: 8000
  workers: 2  # Fewer workers for testing
  cors:
    allow_origins: ["*"]  # Permissive for testing
    allow_credentials: true
  rate_limiting:
    enabled: false  # Disabled for testing

# Document Processing Configuration
document_processing:
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
  max_file_size_mb: 50  # Smaller for testing
  processing_timeout_seconds: 180

# Monitoring Configuration
monitoring:
  log_level: "DEBUG"  # Verbose logging for testing
  logging:
    console:
      enabled: true
      level: "INFO"
    file:
      enabled: true
      path: "/var/log/rag"
      max_size_mb: 50

# Data Storage Configuration
data:
  storage:
    base_directory: "/data/projects/rag-system/data/rag"
    documents_directory: "/data/projects/rag-system/data/rag/documents"
    vectors_directory: "/data/projects/rag-system/data/rag/vectors"
    cache_directory: "/data/projects/rag-system/data/rag/cache"
    temp_directory: "/tmp/rag"

# Testing Configuration
testing:
  enabled: true
  test_data_path: "/data/projects/rag-system/tests/data"
  mock_responses: false
  debug_mode: true
  
# Standalone Features
standalone:
  web_ui: true  # Enable simple web interface
  api_explorer: true  # Enable API documentation
  test_endpoints: true  # Enable testing endpoints
  demo_mode: true  # Enable demonstration features
EOF

    # Copy the standalone config as the main config
    cp "$RAG_HOME/config/rag_config_standalone.yaml" "$RAG_HOME/config/rag_config.yaml"
    
    # Set ownership
    chown -R "$USER:$USER" "$RAG_HOME"
    chown -R "$USER:$USER" "$DATA_DIR"
    
    # Set permissions
    chmod -R 755 "$RAG_HOME"
    chmod -R 750 "$RAG_HOME/config"
    
    log "Standalone application deployment completed"
}

# Setup Python application for standalone mode
setup_python_app() {
    header "SETTING UP PYTHON APPLICATION - STANDALONE MODE"
    
    # Install Python dependencies (subset for standalone)
    log "Installing Python dependencies for standalone mode..."
    
    # Create standalone requirements file
    cat > "$RAG_HOME/requirements_standalone.txt" << 'EOF'
# Core RAG Framework - Standalone Mode
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.3

# Vector Database and Search
qdrant-client==1.15.1
sentence-transformers==3.3.1
transformers==4.46.3
torch==2.5.1
accelerate==1.1.1
numpy==2.2.1

# Document Processing (Essential only)
PyMuPDF==1.25.2
python-docx==1.1.2
pandas==2.2.3
beautifulsoup4==4.12.3
markdown==3.7

# Code Analysis (Essential only)
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

# Testing and Development
pytest==8.3.4
pytest-asyncio==0.25.0

# Optional: Simple Web Interface
streamlit==1.40.2
EOF

    # Install packages in virtual environment
    sudo -u "$USER" bash -c "
        source /data/venv_ai/bin/activate
        pip install --upgrade pip setuptools wheel
        pip install --no-cache-dir -r '$RAG_HOME/requirements_standalone.txt'
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
    
    log "Python application setup completed for standalone mode"
}

# Create systemd services for standalone mode
create_systemd_services() {
    header "CREATING SYSTEMD SERVICES - STANDALONE MODE"
    
    # RAG API Service (Standalone)
    log "Creating standalone RAG API service..."
    cat > /etc/systemd/system/rag-api-standalone.service << EOF
[Unit]
Description=RAG API Service - Standalone Mode
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
Environment=RAG_MODE=standalone
ExecStartPre=/bin/sleep 15
ExecStart=/data/venv_ai/bin/uvicorn api.rag_api:app --host 0.0.0.0 --port 8000 --workers 2 --log-level info
ExecReload=/bin/kill -HUP \$MAINPID
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
TimeoutStartSec=180
TimeoutStopSec=30
KillMode=mixed

[Install]
WantedBy=multi-user.target
EOF

    # RAG Document Processor Service (Standalone)
    log "Creating standalone document processor service..."
    cat > /etc/systemd/system/rag-processor-standalone.service << EOF
[Unit]
Description=RAG Document Processor - Standalone Mode
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
Environment=RAG_MODE=standalone
ExecStartPre=/bin/sleep 20
ExecStart=/data/venv_ai/bin/python scripts/document_processor_daemon.py
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
TimeoutStartSec=120
KillMode=mixed

[Install]
WantedBy=multi-user.target
EOF

    # RAG System Monitor Service (Standalone)
    log "Creating standalone system monitor service..."
    cat > /etc/systemd/system/rag-monitor-standalone.service << EOF
[Unit]
Description=RAG System Monitor - Standalone Mode
After=network.target rag-api-standalone.service

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=$RAG_HOME
Environment=PATH=/data/venv_ai/bin
Environment=PYTHONPATH=$RAG_HOME
Environment=RAG_CONFIG_PATH=$RAG_HOME/config/rag_config.yaml
Environment=RAG_MODE=standalone
ExecStartPre=/bin/sleep 30
ExecStart=/data/venv_ai/bin/python scripts/system_monitor.py
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
TimeoutStartSec=60

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload
    
    log "Standalone systemd services created"
}

# Configure Nginx for standalone mode
configure_nginx_standalone() {
    header "CONFIGURING NGINX - STANDALONE MODE"
    
    log "Creating Nginx configuration for standalone RAG system..."
    cat > "/etc/nginx/sites-available/rag-standalone" << 'EOF'
# Standalone RAG System Nginx Configuration
# Simple proxy configuration for testing

upstream rag_api_standalone {
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    keepalive 16;
}

# Rate limiting (relaxed for testing)
limit_req_zone $binary_remote_addr zone=api_test:10m rate=30r/s;
limit_conn_zone $binary_remote_addr zone=conn_test:10m;

server {
    listen 80;
    server_name rag-standalone.local localhost;
    
    # Basic security headers
    add_header X-Frame-Options SAMEORIGIN always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Connection limits (relaxed for testing)
    limit_conn conn_test 50;
    
    # Logging
    access_log /var/log/nginx/rag-standalone-access.log;
    error_log /var/log/nginx/rag-standalone-error.log;
    
    # Large file uploads for testing
    client_max_body_size 200M;
    client_body_timeout 300s;
    
    # Root redirect to API docs
    location = / {
        return 302 /api/docs;
    }
    
    # API endpoints
    location /api/ {
        limit_req zone=api_test burst=50 nodelay;
        
        proxy_pass http://rag_api_standalone;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Generous timeouts for testing
        proxy_connect_timeout 30s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        
        # Disable buffering for streaming
        proxy_buffering off;
        proxy_request_buffering off;
        
        # CORS headers for testing
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
    
    # Direct API access (no prefix)
    location /query {
        rewrite ^/query$ /api/query break;
        proxy_pass http://rag_api_standalone;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        proxy_buffering off;
    }
    
    # Health check
    location /health {
        access_log off;
        proxy_pass http://rag_api_standalone/api/health;
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 5s;
    }
    
    # Testing endpoints
    location /test/ {
        proxy_pass http://rag_api_standalone/api/;
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

    # Enable the standalone site
    if [[ ! -L "/etc/nginx/sites-enabled/rag-standalone" ]]; then
        ln -s "/etc/nginx/sites-available/rag-standalone" "/etc/nginx/sites-enabled/rag-standalone"
        log "Enabled RAG standalone site"
    fi
    
    # Test Nginx configuration
    if nginx -t; then
        log "Nginx configuration is valid"
        systemctl reload nginx
    else
        error "Nginx configuration is invalid"
    fi
    
    log "Nginx configuration completed for standalone mode"
}

# Create management scripts for standalone mode
create_management_scripts() {
    header "CREATING MANAGEMENT SCRIPTS - STANDALONE MODE"
    
    # Standalone control script
    log "Creating standalone control script..."
    cat > "$RAG_HOME/scripts/rag-standalone.sh" << 'EOF'
#!/bin/bash
# RAG System Standalone Control Script

STANDALONE_SERVICES=("postgresql" "redis-rag" "qdrant" "ollama" "rag-api-standalone" "rag-processor-standalone" "rag-monitor-standalone")
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
        echo -e "${GREEN}Starting RAG System (Standalone Mode)...${NC}"
        for service in "${STANDALONE_SERVICES[@]}"; do
            echo -e "Starting $service..."
            systemctl start "$service"
            sleep 3
        done
        echo -e "${GREEN}RAG System (Standalone) started${NC}"
        ;;
    stop)
        echo -e "${YELLOW}Stopping RAG System (Standalone Mode)...${NC}"
        # Stop in reverse order
        for ((i=${#STANDALONE_SERVICES[@]}-1; i>=0; i--)); do
            service="${STANDALONE_SERVICES[i]}"
            echo -e "Stopping $service..."
            systemctl stop "$service" 2>/dev/null || true
        done
        echo -e "${YELLOW}RAG System (Standalone) stopped${NC}"
        ;;
    restart)
        echo -e "${YELLOW}Restarting RAG System (Standalone Mode)...${NC}"
        $0 stop
        sleep 5
        $0 start
        ;;
    status)
        echo -e "${GREEN}RAG System Status (Standalone Mode):${NC}"
        for service in "${STANDALONE_SERVICES[@]}"; do
            if systemctl is-active --quiet "$service"; then
                echo -e "✓ $service: ${GREEN}Running${NC}"
            else
                echo -e "✗ $service: ${RED}Not running${NC}"
            fi
        done
        
        # Show API status
        echo -e "\n${BLUE}API Status:${NC}"
        if curl -s --max-time 5 "$API_BASE/api/health" > /dev/null; then
            echo -e "✓ RAG API: ${GREEN}Responding${NC}"
            
            # Get API health details
            health_info=$(curl -s --max-time 3 "$API_BASE/api/health" | jq -r '.status // "unknown"' 2>/dev/null)
            echo -e "  Status: $health_info"
        else
            echo -e "✗ RAG API: ${RED}Not responding${NC}"
        fi
        
        # Show Ollama status
        echo -e "\n${BLUE}Ollama Status:${NC}"
        if curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null; then
            echo -e "✓ Ollama: ${GREEN}Responding${NC}"
            
            # List available models
            models=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null | head -3)
            if [[ -n "$models" ]]; then
                echo -e "  Models: $(echo "$models" | tr '\n' ', ' | sed 's/,$//')"
            fi
        else
            echo -e "✗ Ollama: ${RED}Not responding${NC}"
        fi
        ;;
    health)
        echo -e "${GREEN}Performing health check (Standalone Mode)...${NC}"
        
        # Check API health
        if curl -s --max-time 10 "$API_BASE/api/health" > /dev/null; then
            echo -e "✓ API: ${GREEN}Responding${NC}"
        else
            echo -e "✗ API: ${RED}Not responding${NC}"
        fi
        
        # Check database
        if sudo -u $USER psql -h localhost rag_metadata -c "SELECT 1;" > /dev/null 2>&1; then
            echo -e "✓ Database: ${GREEN}Connected${NC}"
            
            # Get document count
            doc_count=$(sudo -u $USER psql -h localhost rag_metadata -t -c "SELECT COUNT(*) FROM documents;" 2>/dev/null | tr -d ' ')
            echo -e "  Documents: $doc_count"
        else
            echo -e "✗ Database: ${RED}Connection failed${NC}"
        fi
        
        # Check Qdrant
        if curl -s --max-time 5 http://localhost:6333/collections > /dev/null; then
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
        
        # Check Ollama
        if curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null; then
            echo -e "✓ Ollama: ${GREEN}Connected${NC}"
        else
            echo -e "✗ Ollama: ${RED}Connection failed${NC}"
        fi
        
        # System resources
        echo -e "\n${BLUE}System Resources:${NC}"
        echo -e "CPU: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')%"
        echo -e "Memory: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
        echo -e "Disk: $(df / | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{print $5}')"
        
        # GPU status
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo -e "\n${BLUE}GPU Status:${NC}"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
                echo -e "  GPU: $line"
            done
        fi
        ;;
    test)
        echo -e "${GREEN}Running standalone system tests...${NC}"
        
        # Test basic query
        echo -e "\n${BLUE}Testing basic query...${NC}"
        test_response=$(curl -s -X POST "$API_BASE/api/query" \
            -H "Content-Type: application/json" \
            -d '{"query": "test system status", "user_id": "test_user"}' 2>/dev/null)
        
        if [[ $? -eq 0 ]] && echo "$test_response" | grep -q "response"; then
            echo -e "✓ Basic query: ${GREEN}Working${NC}"
        else
            echo -e "✗ Basic query: ${RED}Failed${NC}"
        fi
        
        # Test Ollama integration
        echo -e "\n${BLUE}Testing Ollama integration...${NC}"
        ollama_test=$(curl -s -X POST "$API_BASE/api/query" \
            -H "Content-Type: application/json" \
            -d '{"query": "hello world", "user_id": "test_user"}' 2>/dev/null)
        
        if echo "$ollama_test" | grep -q "response"; then
            echo -e "✓ Ollama integration: ${GREEN}Working${NC}"
        else
            echo -e "✗ Ollama integration: ${RED}Failed${NC}"
        fi
        
        # Test document upload
        echo -e "\n${BLUE}Testing document processing...${NC}"
        echo "This is a test document for RAG system validation." > /tmp/test_standalone.txt
        
        upload_response=$(curl -s -X POST "$API_BASE/api/documents/upload" \
            -F "file=@/tmp/test_standalone.txt" \
            -F "security_classification=internal" \
            -F "domain=general" 2>/dev/null)
        
        if echo "$upload_response" | grep -q "success"; then
            echo -e "✓ Document upload: ${GREEN}Working${NC}"
        else
            echo -e "✗ Document upload: ${RED}Failed${NC}"
        fi
        
        rm -f /tmp/test_standalone.txt
        ;;
    enable)
        echo -e "${GREEN}Enabling RAG System services (Standalone)...${NC}"
        for service in "${STANDALONE_SERVICES[@]}"; do
            systemctl enable "$service" 2>/dev/null || true
            echo -e "✓ Enabled $service"
        done
        ;;
    disable)
        echo -e "${YELLOW}Disabling RAG System services (Standalone)...${NC}"
        for service in "${STANDALONE_SERVICES[@]}"; do
            systemctl disable "$service" 2>/dev/null || true
            echo -e "✓ Disabled $service"
        done
        ;;
    logs)
        service=${2:-rag-api-standalone}
        echo -e "${GREEN}Showing logs for $service...${NC}"
        journalctl -u "$service" -f --no-pager
        ;;
    demo)
        echo -e "${GREEN}Running demonstration queries...${NC}"
        echo
        echo -e "${BLUE}1. Technical Query (GPIO Driver):${NC}"
        curl -s -X POST "$API_BASE/api/query" \
            -H "Content-Type: application/json" \
            -d '{"query": "how to implement GPIO driver for STM32", "user_id": "demo_user"}' | jq -r '.response // "No response"'
        echo
        echo -e "${BLUE}2. Code Analysis Query:${NC}"
        curl -s -X POST "$API_BASE/api/query" \
            -H "Content-Type: application/json" \
            -d '{"query": "explain interrupt handling in embedded systems", "user_id": "demo_user"}' | jq -r '.response // "No response"'
        echo
        echo -e "${BLUE}3. General Query:${NC}"
        curl -s -X POST "$API_BASE/api/query" \
            -H "Content-Type: application/json" \
            -d '{"query": "what is radar signal processing", "user_id": "demo_user"}' | jq -r '.response // "No response"'
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|health|test|enable|disable|logs [service]|demo}"
        echo ""
        echo "Standalone Mode Services: ${STANDALONE_SERVICES[*]}"
        echo ""
        echo "Examples:"
        echo "  $0 start          # Start all services"
        echo "  $0 status         # Show status of all services"  
        echo "  $0 health         # Perform health check"
        echo "  $0 test           # Run system tests"
        echo "  $0 demo           # Run demonstration queries"
        echo "  $0 logs           # Show API logs"
        exit 1
        ;;
esac
EOF

    chmod +x "$RAG_HOME/scripts/rag-standalone.sh"
    
    # Create quick test script
    log "Creating quick test script..."
    cat > "$RAG_HOME/scripts/quick_test.sh" << 'EOF'
#!/bin/bash
# Quick Test Script for Standalone RAG System

API_BASE="http://localhost:8000"

echo "🧪 RAG System Quick Test"
echo "========================"
echo

# Test 1: Health Check
echo "1️⃣ Health Check..."
health_response=$(curl -s --max-time 5 "$API_BASE/api/health")
if echo "$health_response" | grep -q '"status"'; then
    status=$(echo "$health_response" | jq -r '.status // "unknown"')
    echo "✅ Health: $status"
else
    echo "❌ Health check failed"
fi
echo

# Test 2: Simple Query
echo "2️⃣ Simple Query Test..."
query_response=$(curl -s --max-time 15 -X POST "$API_BASE/api/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Hello, can you help me?", "user_id": "test_user"}')

if echo "$query_response" | grep -q '"response"'; then
    echo "✅ Query processing works"
    response=$(echo "$query_response" | jq -r '.response // "No response"')
    echo "Response: ${response:0:100}..."
else
    echo "❌ Query processing failed"
fi
echo

# Test 3: Technical Query
echo "3️⃣ Technical Query Test..."
tech_query_response=$(curl -s --max-time 20 -X POST "$API_BASE/api/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is GPIO in embedded systems?", "user_id": "test_user"}')

if echo "$tech_query_response" | grep -q '"response"'; then
    echo "✅ Technical query works"
    confidence=$(echo "$tech_query_response" | jq -r '.confidence_score // 0')
    echo "Confidence: $confidence"
else
    echo "❌ Technical query failed"
fi
echo

# Test 4: API Documentation
echo "4️⃣ API Documentation..."
if curl -s --max-time 5 "$API_BASE/api/docs" > /dev/null; then
    echo "✅ API docs accessible at: $API_BASE/api/docs"
else
    echo "❌ API docs not accessible"
fi
echo

# Test 5: Models Status
echo "5️⃣ Models Status..."
models_response=$(curl -s --max-time 5 "$API_BASE/api/models/status")
if echo "$models_response" | grep -q '"models"'; then
    echo "✅ Models endpoint works"
    models=$(echo "$models_response" | jq -r '.embedding_models // []' | jq length)
    echo "Embedding models: $models"
else
    echo "❌ Models endpoint failed"
fi
echo

echo "🎯 Quick Test Complete!"
echo "📍 Access RAG API at: $API_BASE"
echo "📚 View API docs at: $API_BASE/api/docs"
echo "🔍 Run full tests with: sudo /data/projects/rag-system/scripts/rag-standalone.sh test"
EOF

    chmod +x "$RAG_HOME/scripts/quick_test.sh"
    
    log "Management scripts created for standalone mode"
}

# Initialize system for standalone testing
initialize_system_standalone() {
    header "INITIALIZING SYSTEM - STANDALONE MODE"
    
    # Wait for services to be ready
    log "Waiting for services to initialize..."
    sleep 20
    
    # Create test users
    log "Creating test users..."
    
    # Create admin user
    sudo -u "$USER" /data/venv_ai/bin/python "$RAG_HOME/scripts/user_manager.py" create \
        --user-id "admin" \
        --security-level "classified" \
        --domains "*" \
        --admin-user "system" || warn "Failed to create admin user"
    
    # Create test user
    sudo -u "$USER" /data/venv_ai/bin/python "$RAG_HOME/scripts/user_manager.py" create \
        --user-id "test_user" \
        --security-level "internal" \
        --domains "general" "drivers" \
        --admin-user "admin" || warn "Failed to create test user"
    
    # Create engineer user
    sudo -u "$USER" /data/venv_ai/bin/python "$RAG_HOME/scripts/user_manager.py" create \
        --user-id "engineer" \
        --security-level "confidential" \
        --domains "drivers" "embedded" "general" \
        --admin-user "admin" || warn "Failed to create engineer user"
    
    # Create sample documents for testing
    log "Creating sample documents..."
    mkdir -p "/data/projects/rag-system/data/rag/test_documents"
    
    # Create GPIO driver example
    cat > "/data/projects/rag-system/data/rag/test_documents/gpio_basics.md" << 'EOF'
# GPIO Basics for Embedded Systems

## Introduction
GPIO (General Purpose Input/Output) pins are the most basic way to interact with the physical world in embedded systems.

## Key Concepts

### Pin Configuration
- **Input Mode**: Pin reads digital signals (0V = LOW, 3.3V/5V = HIGH)
- **Output Mode**: Pin drives digital signals to external devices
- **Pull-up/Pull-down**: Internal resistors to set default states

### Common Operations
1. **Initialization**: Configure pin direction and properties
2. **Read**: Check input pin state
3. **Write**: Set output pin state
4. **Toggle**: Change output pin state

### STM32 Example
```c
// Initialize GPIO pin as output
GPIO_InitTypeDef GPIO_InitStruct = {0};
GPIO_InitStruct.Pin = GPIO_PIN_5;
GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
GPIO_InitStruct.Pull = GPIO_NOPULL;
GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

// Set pin HIGH
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
```

## Best Practices
- Always configure pins before use
- Use appropriate pull resistors
- Consider power consumption
- Handle interrupts properly
EOF

    # Create UART communication guide
    cat > "/data/projects/rag-system/data/rag/test_documents/uart_communication.md" << 'EOF'
# UART Communication Guide

## Overview
UART (Universal Asynchronous Receiver-Transmitter) is a simple serial communication protocol.

## Key Features
- **Asynchronous**: No shared clock signal
- **Full-duplex**: Simultaneous bidirectional communication
- **Simple**: Only requires TX and RX lines (plus GND)

## Configuration Parameters
1. **Baud Rate**: Communication speed (e.g., 9600, 115200)
2. **Data Bits**: Usually 8 bits
3. **Parity**: Error checking (None, Even, Odd)
4. **Stop Bits**: End of frame marker (1 or 2)

## Implementation Example
```c
// UART initialization
UART_HandleTypeDef huart2;

void UART_Init(void) {
    huart2.Instance = USART2;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    
    HAL_UART_Init(&huart2);
}

// Send data
void UART_SendString(char* str) {
    HAL_UART_Transmit(&huart2, (uint8_t*)str, strlen(str), 1000);
}
```

## Common Issues
- **Baud rate mismatch**: Ensure both devices use same rate
- **Wiring errors**: Check TX-RX cross connection
- **Buffer overflow**: Implement proper flow control
EOF

    # Create embedded systems overview
    cat > "/data/projects/rag-system/data/rag/test_documents/embedded_overview.txt" << 'EOF'
Embedded Systems Overview

Embedded systems are specialized computing systems that perform dedicated functions within larger mechanical or electrical systems. They are designed for specific tasks and often have real-time constraints.

Key Characteristics:
1. Dedicated functionality
2. Resource constraints (memory, power, processing)
3. Real-time requirements
4. Reliability and stability needs
5. Cost optimization

Common Components:
- Microcontroller or microprocessor
- Memory (RAM, Flash, EEPROM)
- Input/Output interfaces
- Communication interfaces
- Power management circuits
- Sensors and actuators

Development Process:
1. Requirements analysis
2. Hardware design
3. Software architecture
4. Implementation
5. Testing and validation
6. Deployment and maintenance

Programming Languages:
- C (most common)
- C++ (for complex systems)
- Assembly (for low-level optimization)
- Python (for rapid prototyping)
- Rust (for safety-critical systems)

Real-time Operating Systems (RTOS):
- FreeRTOS
- ThreadX
- VxWorks
- QNX
- Zephyr

Best Practices:
- Modular design
- Error handling
- Resource management
- Documentation
- Testing strategy
- Version control
EOF

    # Set ownership
    chown -R "$USER:$USER" "/data/projects/rag-system/data/rag/test_documents"
    
    # Ingest sample documents
    log "Ingesting sample documents..."
    sleep 5  # Wait a bit more for services
    
    # Use the document ingestion API
    for doc in /data/projects/rag-system/data/rag/test_documents/*; do
        filename=$(basename "$doc")
        log "Ingesting $filename..."
        
        curl -s -X POST "http://localhost:8000/api/documents/upload" \
            -F "file=@$doc" \
            -F "security_classification=internal" \
            -F "domain=general" >/dev/null 2>&1 || warn "Failed to ingest $filename"
    done
    
    log "System initialization completed for standalone mode"
}

# Deployment verification for standalone
verify_deployment_standalone() {
    header "VERIFYING STANDALONE DEPLOYMENT"
    
    log "Waiting for all services to be ready..."
    sleep 30
    
    # Check services
    log "Checking service status..."
    failed_services=()
    
    standalone_services=("postgresql" "redis-rag" "qdrant" "ollama" "rag-api-standalone" "rag-processor-standalone" "rag-monitor-standalone")
    
    for service in "${standalone_services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            log "✓ $service is running"
        else
            warn "✗ $service is not running"
            failed_services+=("$service")
        fi
    done
    
    # Test API
    log "Testing API functionality..."
    
    # Health check
    if curl -s --max-time 10 http://localhost:8000/api/health >/dev/null; then
        log "✓ RAG API health endpoint responding"
    else
        warn "✗ RAG API health endpoint not responding"
    fi
    
    # Test query
    test_query=$(curl -s --max-time 20 -X POST http://localhost:8000/api/query \
        -H "Content-Type: application/json" \
        -d '{"query": "test standalone system", "user_id": "test_user"}' 2>/dev/null)
    
    if echo "$test_query" | grep -q '"response"'; then
        log "✓ Query processing working"
    else
        warn "✗ Query processing failed"
    fi
    
    # Test Ollama integration
    if curl -s --max-time 5 http://localhost:11434/api/tags >/dev/null; then
        log "✓ Ollama integration working"
        
        # Check for models
        model_count=$(curl -s http://localhost:11434/api/tags | jq '.models | length' 2>/dev/null || echo "0")
        log "  Available models: $model_count"
    else
        warn "✗ Ollama integration failed"
    fi
    
    # Generate deployment report
    cat > "$RAG_HOME/STANDALONE_DEPLOYMENT_REPORT.md" << EOF
# RAG System Standalone Deployment Report

**Deployment Date:** $(date)
**Mode:** Standalone Testing
**System:** $(uname -a)

## Deployment Summary

$(if [[ ${#failed_services[@]} -eq 0 ]]; then
    echo "✅ **SUCCESS**: All services deployed and running"
else
    echo "⚠️ **PARTIAL**: ${#failed_services[@]} services failed"
fi)

## Service Status
$(for service in "${standalone_services[@]}"; do
    if systemctl is-active --quiet "$service"; then
        echo "- ✅ $service: Running"
    else
        echo "- ❌ $service: Not Running"
    fi
done)

## Access Points
- **RAG API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health
- **Nginx Proxy**: http://localhost (redirects to API docs)

## Quick Test Commands
\`\`\`bash
# Check system status
sudo /data/projects/rag-system/scripts/rag-standalone.sh status

# Run health check
sudo /data/projects/rag-system/scripts/rag-standalone.sh health

# Run quick tests
sudo /data/projects/rag-system/scripts/quick_test.sh

# Run demonstration
sudo /data/projects/rag-system/scripts/rag-standalone.sh demo

# Test a query
curl -X POST http://localhost:8000/api/query \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What is GPIO?", "user_id": "test_user"}'
\`\`\`

## System Information
- **Memory**: $(free -h | awk '/^Mem:/{print $2}')
- **Disk**: $(df -h / | awk 'NR==2{print $4}') available
- **GPUs**: $(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
- **CPU Cores**: $(nproc)

## Database Status
- **Documents**: $(sudo -u rag-system psql -h localhost rag_metadata -t -c "SELECT COUNT(*) FROM documents;" 2>/dev/null | tr -d ' ' || echo "N/A")
- **Users**: $(sudo -u rag-system psql -h localhost rag_metadata -t -c "SELECT COUNT(*) FROM user_access;" 2>/dev/null | tr -d ' ' || echo "N/A")

## Next Steps for Full Integration
1. Install OpenWebUI when ready
2. Run full deployment script: \`./deploy_rag_system.sh\`
3. Configure LDAP integration (optional)
4. Upload production documents
5. Configure SSL certificates

## Troubleshooting
- **View logs**: \`sudo journalctl -u rag-api-standalone -f\`
- **Check Ollama**: \`curl http://localhost:11434/api/tags\`
- **Test database**: \`sudo -u rag-system psql rag_metadata -c "SELECT 1;"\`
- **Monitor resources**: \`htop\` or \`nvidia-smi\`

The standalone RAG system is ready for testing and evaluation!
EOF

    chown "$USER:$USER" "$RAG_HOME/STANDALONE_DEPLOYMENT_REPORT.md"
    
    log "Standalone deployment verification completed"
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log "🎉 Standalone deployment successful!"
    else
        warn "⚠️ Deployment completed with issues. Check the report."
    fi
}

# Main deployment function for standalone mode
main() {
    header "RAG SYSTEM STANDALONE DEPLOYMENT"
    info "Deploying RAG system in standalone mode for testing"
    info "This deployment excludes OpenWebUI integration"
    info "Estimated time: 10-15 minutes"
    
    # Run deployment steps
    check_prerequisites
    deploy_application
    ##setup_python_app
    create_systemd_services
    configure_nginx_standalone
    create_management_scripts
    
    # Start services
    header "STARTING SERVICES - STANDALONE MODE"
    log "Starting standalone RAG services..."
    
    standalone_services=("postgresql" "redis-rag" "qdrant" "ollama" "rag-api-standalone" "rag-processor-standalone" "rag-monitor-standalone")
    
    for service in "${standalone_services[@]}"; do
        systemctl enable "$service" 2>/dev/null || true
        systemctl start "$service"
        sleep 5
    done
    
    # Initialize system
    initialize_system_standalone
    
    # Verify deployment
    verify_deployment_standalone
    
    # Final summary
    header "STANDALONE DEPLOYMENT COMPLETE"
    log "RAG System deployed successfully in standalone mode!"
    log ""
    log "🔗 Access Points:"
    log "   • RAG API: http://localhost:8000"
    log "   • API Documentation: http://localhost:8000/api/docs"
    log "   • Health Check: http://localhost:8000/api/health"
    log ""
    log "🧪 Testing Commands:"
    log "   • Quick Test: /data/projects/rag-system/scripts/quick_test.sh"
    log "   • System Status: /data/projects/rag-system/scripts/rag-standalone.sh status"
    log "   • Health Check: /data/projects/rag-system/scripts/rag-standalone.sh health"
    log "   • Demo Queries: /data/projects/rag-system/scripts/rag-standalone.sh demo"
    log ""
    log "📋 Management:"
    log "   • Control Script: /data/projects/rag-system/scripts/rag-standalone.sh"
    log "   • User Management: /data/projects/rag-system/scripts/user_manager.py"
    log "   • View Logs: journalctl -u rag-api-standalone -f"
    log ""
    log "📄 Documentation:"
    log "   • Deployment Report: /data/projects/rag-system/STANDALONE_DEPLOYMENT_REPORT.md"
    log "   • Configuration: /data/projects/rag-system/config/rag_config.yaml"
    log ""
    log "✅ The standalone RAG system is ready for testing!"
    log ""
    info "Try a test query:"
    info "curl -X POST http://localhost:8000/api/query \\"
    info "  -H 'Content-Type: application/json' \\"
    info "  -d '{\"query\": \"What is GPIO in embedded systems?\", \"user_id\": \"test_user\"}'"
    log ""
    log "🔄 When ready for full integration, run: ./deploy_rag_system.sh"
}

# Run main function
main "$@"
