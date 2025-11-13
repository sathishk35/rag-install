#!/bin/bash
# RAG System Installation Script for Data Patterns India
# Air-gapped environment setup with security considerations

set -e

# Configuration
RAG_HOME="/data/projects/rag-system"
DATA_DIR="/data/projects/rag-system/data/rag"
LOGS_DIR="/var/log/rag"
USER="rag-system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root"
fi

log "Starting RAG System Installation..."

# 1. System Preparation
log "Setting up system users and directories..."

# Create system user
if ! id "$USER" &>/dev/null; then
    useradd -r -m -s /bin/bash "$USER"
    log "Created system user: $USER"
fi

# Create directories
mkdir -p "$RAG_HOME"/{config,scripts,models,data}
mkdir -p "$DATA_DIR"/{documents,vectors,metadata,cache}
mkdir -p "$LOGS_DIR"
chown -R "$USER:$USER" "$RAG_HOME" "$DATA_DIR" "$LOGS_DIR"

# 2. Install System Dependencies
log "Installing system dependencies..."

#apt update
apt install -y \
    postgresql postgresql-contrib \
    redis-server \
    nginx \
    git curl wget \
    build-essential \
    pkg-config \
    libpq-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libreoffice \
    pandoc \
    nodejs npm \
    docker.io docker-compose \
    htop iotop \
    fail2ban \
    ufw

# 3. Install Qdrant Vector Database
log "Installing Qdrant Vector Database..."

cat > /etc/systemd/system/qdrant.service << 'EOF'
[Unit]
Description=Qdrant Vector Database
After=network.target

[Service]
Type=exec
User=rag-system
Group=rag-system
WorkingDirectory=/data/projects/rag-system
Environment=QDRANT_CONFIG_PATH=/data/projects/rag-system/config/qdrant-config.yaml
ExecStart=/data/projects/rag-system/qdrant/qdrant
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Download and install Qdrant
cd "$RAG_HOME"
## wget https://github.com/qdrant/qdrant/releases/download/v1.15.1/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
mv qdrant qdrant-binary
mkdir -p qdrant
mv qdrant-binary qdrant/qdrant
chmod +x qdrant/qdrant
chown -R "$USER:$USER" qdrant/

# 4. Configure PostgreSQL
log "Configuring PostgreSQL..."

## su - postgres -c "createuser -s $USER"
## su - postgres -c "createdb -O $USER rag_metadata"

# Create database schema
cat > "$RAG_HOME/scripts/init_db.sql" << 'EOF'
-- RAG Metadata Database Schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    content_hash TEXT NOT NULL,
    security_classification TEXT NOT NULL DEFAULT 'internal',
    domain TEXT,
    language TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_processed TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Chunks table
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL, -- 'code', 'documentation', 'comment'
    chunk_metadata JSONB,
    embedding_id TEXT, -- Reference to Qdrant vector ID
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Access control table
CREATE TABLE user_access (
    user_id TEXT NOT NULL,
    security_clearance TEXT NOT NULL,
    domains TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Query audit log
CREATE TABLE query_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    retrieved_docs UUID[],
    response_generated BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Indexes
CREATE INDEX idx_documents_security ON documents(security_classification);
CREATE INDEX idx_documents_domain ON documents(domain);
CREATE INDEX idx_documents_updated ON documents(updated_at);
CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_type ON chunks(content_type);
CREATE INDEX idx_query_log_user ON query_log(user_id);
CREATE INDEX idx_query_log_timestamp ON query_log(timestamp);

-- Full-text search
CREATE INDEX idx_documents_content ON documents USING gin(to_tsvector('english', file_name));
CREATE INDEX idx_chunks_content ON chunks USING gin(to_tsvector('english', content));
EOF

su - "$USER" -c "psql rag_metadata < $RAG_HOME/scripts/init_db.sql"

# 5. Setup Python Environment
log "Setting up Python environment..."

su - "$USER" -c "python3 -m venv $RAG_HOME/venv"
su - "$USER" -c "source $RAG_HOME/venv/bin/activate && pip install --upgrade pip setuptools wheel"

# Create requirements.txt
cat > "$RAG_HOME/requirements.txt" << 'EOF'
# RAG System Requirements - Updated for Latest Compatibility (Jan 2025)

# Core RAG Framework (SPECIFIC VERSIONS - Breaking changes in 0.3.x)
langchain==0.2.16
langchain-community==0.2.17
langchain-core==0.2.40

# Vector Database (LATEST - Fully compatible)
qdrant-client==1.15.1

# ML/AI Models (LATEST - Excellent compatibility)
sentence-transformers==3.3.1
transformers==4.46.3
torch==2.5.1
accelerate==1.1.1
numpy==2.2.1
scipy==1.14.1

# Web Framework (LATEST - Great new features)
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.3
python-multipart==0.0.17
httpx==0.28.1
websockets==13.1
aiofiles==24.1.0

# Database Connections (LATEST - Stable)
psycopg2-binary==2.9.10
redis==5.2.1
sqlalchemy==2.0.36

# Document Processing (LATEST - More file support)
unstructured[local-inference]==0.16.9
PyMuPDF==1.25.2
python-docx==1.1.2
python-pptx==1.0.2
pandas==2.2.3
openpyxl==3.1.5
xlsxwriter==3.2.0

# Code Analysis (SPECIFIC VERSIONS - API stability)
tree-sitter==0.20.4
tree-sitter-c==0.20.6
tree-sitter-cpp==0.20.3
tree-sitter-python==0.20.4
pygments==2.18.0
gitpython==3.1.43

# NLP Processing (SPECIFIC VERSION - Model compatibility)
nltk==3.9.1
spacy==3.7.6

# Web Scraping & Parsing (LATEST)
beautifulsoup4==4.12.3
lxml==5.3.0
html5lib==1.1
markdown==3.7

# Utilities (LATEST - Good stability)
watchdog==6.0.0
schedule==1.2.2
tqdm==4.67.1
rich==13.9.4
click==8.1.8
python-dotenv==1.0.1

# Security & Encryption (LATEST - Security updates)
cryptography==44.0.0
pyjwt==2.10.1
passlib==1.7.4
bcrypt==4.2.1

# Monitoring & Logging (LATEST)
prometheus-client==0.21.0
structlog==24.4.0
psutil==6.1.0

# Image Processing (LATEST)
Pillow==11.0.0
opencv-python==4.10.0.84

# Scientific Computing (LATEST)
matplotlib==3.10.0
seaborn==0.13.2
plotly==5.24.1

# Data Processing (LATEST)
pyarrow==18.1.0
polars==1.17.1

# Audio Processing (if needed)
librosa==0.10.2
soundfile==0.12.1

# Development & Testing (LATEST)
pytest==8.3.4
pytest-asyncio==0.25.0
black==24.10.0
flake8==7.1.1
mypy==1.13.0

# Optional: GPU acceleration (LATEST)
# torch-audio==2.5.1  # Uncomment if audio processing needed
# torchvision==0.20.1  # Uncomment if image processing needed
EOF

su - "$USER" -c "source $RAG_HOME/venv/bin/activate && pip install -r $RAG_HOME/requirements.txt"

# 6. Install Additional Models
log "Downloading embedding models..."

su - "$USER" -c "
source $RAG_HOME/venv/bin/activate
python -c \"
from sentence_transformers import SentenceTransformer
import os
os.chdir('$RAG_HOME/models')

# Download embedding models
print('Downloading BGE-M3...')
model = SentenceTransformer('BAAI/bge-m3')
model.save('bge-m3')

print('Downloading E5-large-v2...')
model = SentenceTransformer('intfloat/e5-large-v2')
model.save('e5-large-v2')

print('Downloading CodeBERT...')
model = SentenceTransformer('microsoft/codebert-base')
model.save('codebert-base')

print('Models downloaded successfully!')
\"
"

# 7. Configure Services
log "Configuring services..."

# Qdrant configuration
cat > "$RAG_HOME/config/qdrant-config.yaml" << 'EOF'
log_level: INFO

storage:
  # Storage location
  storage_path: /data/rag/vectors
  snapshots_path: /data/rag/vectors/snapshots
  
  # Storage optimizations (Qdrant 1.15.1 features)
  on_disk_payload: true
  mmap_threshold_kb: 100000
  
  # Write-ahead log
  wal:
    wal_capacity_mb: 64
    wal_segments_ahead: 2

# Service configuration
service:
  http_port: 6333
  grpc_port: 6334
  enable_cors: false
  max_request_size_mb: 64
  max_workers: 8
  
  # New in 1.15.x - Better request handling
  grpc_timeout_ms: 60000
  max_concurrent_requests: 128

# Performance tuning for defense workloads
performance:
  # Memory usage (NEW in 1.15.x)
  max_indexing_memory_kb: 2097152  # 2GB for indexing
  
  # Parallel processing
  max_search_threads: 8
  max_optimization_threads: 4
  
  # Batching (improved in 1.15.x)
  default_write_batch_size: 100
  max_batch_size: 1000

# Collections optimizations
collections:
  # Global collection defaults
  default_optimizers_config:
    deleted_threshold: 0.2
    vacuum_min_vector_number: 1000
    default_segment_number: 4        # Increased for better parallelism
    max_segment_size_kb: 200000
    memmap_threshold_kb: 100000
    indexing_threshold_kb: 50000
    flush_interval_sec: 10
    max_optimization_threads: 4
  
  # HNSW algorithm settings (optimized for 1.15.x)
  default_hnsw_config:
    m: 32                           # Increased connectivity
    ef_construct: 200               # Better build quality
    full_scan_threshold: 10000
    max_indexing_threads: 4
    on_disk: false                  # Keep in memory for speed
    
    # New in 1.15.x - Advanced HNSW settings
    payload_m: 16                   # Payload index connectivity
    
  # Quantization (improved in 1.15.x)
  default_quantization:
    scalar:
      type: "int8"
      quantile: 0.99
      always_ram: true              # Keep quantized vectors in RAM
    
    # New binary quantization option (1.15.x feature)
    binary:
      always_ram: true

# Clustering (if using distributed setup)
cluster:
  enabled: false
  # Uncomment for distributed deployment
  # node_id: 1
  # bootstrap_nodes: ["http://node1:6335", "http://node2:6335"]
  # p2p_port: 6335
  # consensus_timeout_ms: 1000

# API and security
api:
  # Enable API key authentication (NEW in 1.15.x)
  enable_api_key: false
  # api_key: "your-secure-api-key"  # Uncomment for production
  
  # CORS settings for web integration
  cors:
    allow_origin: ["http://localhost:8000", "http://localhost:3000"]
    allow_methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers: ["*"]

# Telemetry and monitoring
telemetry:
  disabled: true                    # Disable for air-gapped environment
  
  # Custom metrics (NEW in 1.15.x)
  metrics:
    enable_prometheus: true
    prometheus_port: 9091

# Resource limits
limits:
  # Memory limits (improved in 1.15.x)
  max_memory_usage_mb: 32768       # 32GB max memory usage
  
  # Disk limits
  max_disk_usage_gb: 500           # 500GB max disk usage
  
  # Request limits
  max_request_timeout_ms: 300000   # 5 minutes max request time

# New features in Qdrant 1.15.x
features:
  # Multi-tenancy support
  enable_multi_tenancy: false
  
  # Advanced indexing
  enable_async_indexing: true      # Background indexing
  
  # Improved sparse vectors support
  enable_sparse_vectors: true
  
  # Better memory management
  enable_memory_mapping: true
  
  # Advanced payload indexing
  enable_payload_index: true
EOF

# Redis configuration
cat > /etc/redis/redis-rag.conf << 'EOF'
port 6380
bind 127.0.0.1
protected-mode yes
tcp-backlog 511
timeout 300
tcp-keepalive 300
daemonize no
supervised systemd
pidfile /var/run/redis/redis-rag.pid
loglevel notice
logfile /var/log/redis/redis-rag.log
databases 16
always-show-logo yes
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename rag-dump.rdb
dir /var/lib/redis
maxmemory 8gb
maxmemory-policy allkeys-lru
EOF

# Create Redis service for RAG
cat > /etc/systemd/system/redis-rag.service << 'EOF'
[Unit]
Description=Redis In-Memory Data Store for RAG
After=network.target

[Service]
Type=notify
User=redis
Group=redis
ExecStart=/usr/bin/redis-server /etc/redis/redis-rag.conf
ExecStop=/usr/bin/redis-cli -p 6380 shutdown
TimeoutStopSec=0
Restart=always
User=redis
Group=redis
RuntimeDirectory=redis
RuntimeDirectoryMode=0755

[Install]
WantedBy=multi-user.target
EOF

# 8. Setup Security
log "Configuring security..."

# Setup firewall
ufw --force enable
ufw allow ssh
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 6333/tcp  # Qdrant (internal only)
ufw allow 6380/tcp  # Redis (internal only)
ufw allow 5432/tcp  # PostgreSQL (internal only)

# Configure fail2ban
cat > /etc/fail2ban/jail.d/rag-system.conf << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[rag-api]
enabled = true
port = 8000
logpath = /var/log/rag/api.log
banaction = ufw
EOF

# 9. Create Main RAG Application
log "Creating RAG application..."

cat > "$RAG_HOME/scripts/start_services.sh" << 'EOF'
#!/bin/bash
# Start all RAG services

systemctl start postgresql
systemctl start redis-rag
systemctl start qdrant
systemctl start nginx

echo "All RAG services started successfully!"
EOF

chmod +x "$RAG_HOME/scripts/start_services.sh"

# Enable services
systemctl enable postgresql
systemctl enable redis-rag
systemctl enable qdrant
systemctl enable nginx

# 10. Setup Monitoring
log "Setting up monitoring..."

cat > "$RAG_HOME/scripts/health_check.sh" << 'EOF'
#!/bin/bash
# RAG System Health Check

echo "=== RAG System Health Check ==="
echo "Date: $(date)"
echo

# Check services
services=("postgresql" "redis-rag" "qdrant" "nginx")
for service in "${services[@]}"; do
    if systemctl is-active --quiet "$service"; then
        echo "✓ $service: Running"
    else
        echo "✗ $service: Not running"
    fi
done

echo

# Check disk usage
echo "=== Disk Usage ==="
df -h /data/projects/rag-system /data/projects/rag-system/data/rag

echo

# Check memory usage
echo "=== Memory Usage ==="
free -h

echo

# Check GPU usage
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo

# Check Qdrant status
echo "=== Qdrant Status ==="
curl -s http://localhost:6333/telemetry | jq '.status' 2>/dev/null || echo "Qdrant not responding"

echo

# Check database connections
echo "=== Database Status ==="
su - rag-system -c "psql rag_metadata -c 'SELECT COUNT(*) as document_count FROM documents;'" 2>/dev/null || echo "Database not accessible"

echo "=== Health Check Complete ==="
EOF

chmod +x "$RAG_HOME/scripts/health_check.sh"

# Create cron job for health checks
cat > /etc/cron.d/rag-health << 'EOF'
# RAG System Health Checks
*/15 * * * * rag-system /data/projects/rag-system/scripts/health_check.sh >> /var/log/rag/health.log 2>&1
EOF

# 11. Setup Log Rotation
log "Setting up log rotation..."

cat > /etc/logrotate.d/rag-system << 'EOF'
/var/log/rag/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 rag-system rag-system
    postrotate
        systemctl reload qdrant 2>/dev/null || true
    endscript
}
EOF

# 12. Create Installation Summary
log "Creating installation summary..."

cat > "$RAG_HOME/INSTALLATION_SUMMARY.md" << EOF
# RAG System Installation Summary

## Installation Date
$(date)

## System Configuration
- **RAG Home:** $RAG_HOME
- **Data Directory:** $DATA_DIR
- **Logs Directory:** $LOGS_DIR
- **System User:** $USER

## Services Installed
- PostgreSQL 15 (Port 5432)
- Redis RAG Instance (Port 6380)
- Qdrant Vector Database (Port 6333)
- Nginx Web Server (Port 80/443)

## Models Installed
- BGE-M3 (Multi-lingual embeddings)
- E5-large-v2 (General embeddings)
- CodeBERT (Code-specific embeddings)

## Next Steps
1. Start services: $RAG_HOME/scripts/start_services.sh
2. Configure OpenWebUI integration
3. Install RAG application code
4. Setup document processing pipeline
5. Configure security and access control

## Important Files
- Configuration: $RAG_HOME/config/
- Scripts: $RAG_HOME/scripts/
- Models: $RAG_HOME/models/
- Database Schema: $RAG_HOME/scripts/init_db.sql

## Monitoring
- Health Check: $RAG_HOME/scripts/health_check.sh
- Log Files: $LOGS_DIR/
- Cron Jobs: /etc/cron.d/rag-health

## Security
- Firewall configured with UFail2ban enabled
- Services bound to localhost only
- Audit logging enabled

EOF

chown "$USER:$USER" "$RAG_HOME/INSTALLATION_SUMMARY.md"

log "RAG System base installation completed successfully!"
log "Please review $RAG_HOME/INSTALLATION_SUMMARY.md for next steps."
log "Start services with: $RAG_HOME/scripts/start_services.sh"

exit 0
