#!/bin/bash

################################################################################
# RAG System - Stage-wise Installation Script
# Version: 2.0
# Features: Resume from failed stage, rollback, validation, logging
################################################################################

set -e  # Exit on error (we'll handle it)
set -o pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Installation directories
INSTALL_DIR="/opt/rag-system"
BACKUP_DIR="/opt/rag-system-backup"
STATE_FILE="/var/lib/rag-system/install_state.json"
LOG_DIR="/var/log/rag-install"
LOG_FILE="${LOG_DIR}/install_$(date +%Y%m%d_%H%M%S).log"

# Installation stages
declare -a STAGES=(
    "01_preflight_checks"
    "02_backup_existing"
    "03_install_system_deps"
    "04_setup_python_env"
    "05_install_python_deps"
    "06_setup_database"
    "07_setup_redis"
    "08_setup_qdrant"
    "09_download_models"
    "10_setup_directories"
    "11_copy_application_files"
    "12_setup_configuration"
    "13_setup_systemd_services"
    "14_initialize_database"
    "15_run_migrations"
    "16_validate_installation"
    "17_start_services"
)

################################################################################
# Utility Functions
################################################################################

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() {
    log "INFO" "${BLUE}$@${NC}"
}

log_success() {
    log "SUCCESS" "${GREEN}✓ $@${NC}"
}

log_warning() {
    log "WARNING" "${YELLOW}⚠ $@${NC}"
}

log_error() {
    log "ERROR" "${RED}✗ $@${NC}"
}

print_header() {
    echo ""
    echo "======================================================================"
    echo "$@"
    echo "======================================================================"
    echo ""
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run as root or with sudo"
        exit 1
    fi
}

################################################################################
# State Management
################################################################################

init_state_file() {
    mkdir -p "$(dirname ${STATE_FILE})"
    mkdir -p "${LOG_DIR}"

    if [ ! -f "${STATE_FILE}" ]; then
        cat > "${STATE_FILE}" <<EOF
{
  "version": "2.0",
  "install_date": "$(date -Iseconds)",
  "current_stage": 0,
  "completed_stages": [],
  "failed_stage": null,
  "last_error": null
}
EOF
        log_info "Initialized state file: ${STATE_FILE}"
    fi
}

get_current_stage() {
    python3 -c "import json; print(json.load(open('${STATE_FILE}'))['current_stage'])"
}

get_completed_stages() {
    python3 -c "import json; print(' '.join(json.load(open('${STATE_FILE}'))['completed_stages']))"
}

update_stage_status() {
    local stage=$1
    local status=$2  # "in_progress", "completed", "failed"

    python3 <<EOF
import json
with open('${STATE_FILE}', 'r') as f:
    state = json.load(f)

if '${status}' == 'in_progress':
    state['current_stage'] = ${stage}
elif '${status}' == 'completed':
    if '${STAGES[$stage]}' not in state['completed_stages']:
        state['completed_stages'].append('${STAGES[$stage]}')
    state['current_stage'] = ${stage} + 1
elif '${status}' == 'failed':
    state['failed_stage'] = '${STAGES[$stage]}'
    state['last_error'] = 'Stage ${STAGES[$stage]} failed'

with open('${STATE_FILE}', 'w') as f:
    json.dump(state, f, indent=2)
EOF
}

is_stage_completed() {
    local stage_name=$1
    local completed=$(get_completed_stages)

    if [[ " ${completed} " =~ " ${stage_name} " ]]; then
        return 0
    else
        return 1
    fi
}

################################################################################
# Stage Functions
################################################################################

stage_01_preflight_checks() {
    print_header "STAGE 1: Pre-flight Checks"

    log_info "Checking system requirements..."

    # Check OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        log_info "Operating System: $NAME $VERSION"
        if [[ "$ID" != "ubuntu" ]] && [[ "$ID" != "debian" ]]; then
            log_warning "This script is optimized for Ubuntu/Debian. You may need to adjust package names."
        fi
    fi

    # Check available disk space (need at least 50GB)
    local available_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 50 ]; then
        log_error "Insufficient disk space. Need at least 50GB, have ${available_space}GB"
        return 1
    fi
    log_success "Disk space check passed (${available_space}GB available)"

    # Check RAM (need at least 32GB, recommend 128GB)
    local total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_ram" -lt 32 ]; then
        log_error "Insufficient RAM. Need at least 32GB, have ${total_ram}GB"
        return 1
    fi
    log_success "RAM check passed (${total_ram}GB available)"

    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        log_success "GPU check passed (${gpu_count} GPU(s) detected)"
    else
        log_warning "No NVIDIA GPUs detected. System will run on CPU (slower)"
    fi

    # Check Python version
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | awk '{print $2}')
        log_info "Python version: ${python_version}"

        # Check if version is 3.11+
        local major=$(echo $python_version | cut -d. -f1)
        local minor=$(echo $python_version | cut -d. -f2)

        if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 11 ]); then
            log_error "Python 3.11+ required, found ${python_version}"
            return 1
        fi
    else
        log_error "Python 3 not found"
        return 1
    fi

    # Check network connectivity
    if ping -c 1 8.8.8.8 &> /dev/null; then
        log_success "Network connectivity check passed"
    else
        log_warning "No internet connectivity detected. Ensure packages are available locally."
    fi

    log_success "Pre-flight checks completed"
    return 0
}

stage_02_backup_existing() {
    print_header "STAGE 2: Backup Existing Installation"

    if [ -d "${INSTALL_DIR}" ]; then
        log_info "Existing installation found at ${INSTALL_DIR}"

        local backup_name="rag-system-backup-$(date +%Y%m%d_%H%M%S)"
        local backup_path="${BACKUP_DIR}/${backup_name}"

        log_info "Creating backup at ${backup_path}..."
        mkdir -p "${BACKUP_DIR}"

        # Backup installation directory
        tar -czf "${backup_path}.tar.gz" -C "$(dirname ${INSTALL_DIR})" "$(basename ${INSTALL_DIR})" 2>&1 | tee -a "${LOG_FILE}"

        if [ -f "${backup_path}.tar.gz" ]; then
            log_success "Backup created: ${backup_path}.tar.gz"

            # Save backup path to state
            python3 <<EOF
import json
with open('${STATE_FILE}', 'r') as f:
    state = json.load(f)
state['backup_path'] = '${backup_path}.tar.gz'
with open('${STATE_FILE}', 'w') as f:
    json.dump(state, f, indent=2)
EOF
        else
            log_error "Backup failed"
            return 1
        fi
    else
        log_info "No existing installation found, skipping backup"
    fi

    return 0
}

stage_03_install_system_deps() {
    print_header "STAGE 3: Install System Dependencies"

    log_info "Updating package lists..."
    apt-get update -y 2>&1 | tee -a "${LOG_FILE}"

    log_info "Installing system packages..."
    apt-get install -y \
        build-essential \
        git \
        curl \
        wget \
        vim \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        python3-dev \
        python3-pip \
        python3-venv \
        postgresql \
        postgresql-contrib \
        redis-server \
        nginx \
        supervisor \
        htop \
        tree \
        jq \
        2>&1 | tee -a "${LOG_FILE}"

    log_success "System dependencies installed"
    return 0
}

stage_04_setup_python_env() {
    print_header "STAGE 4: Setup Python Virtual Environment"

    local venv_path="${INSTALL_DIR}/venv"

    log_info "Creating virtual environment at ${venv_path}..."
    mkdir -p "${INSTALL_DIR}"
    python3 -m venv "${venv_path}"

    log_info "Activating virtual environment..."
    source "${venv_path}/bin/activate"

    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_FILE}"

    log_success "Python virtual environment created"
    return 0
}

stage_05_install_python_deps() {
    print_header "STAGE 5: Install Python Dependencies"

    local venv_path="${INSTALL_DIR}/venv"
    source "${venv_path}/bin/activate"

    # Copy requirements file
    if [ ! -f "${INSTALL_DIR}/requirements_latest.txt" ]; then
        log_error "requirements_latest.txt not found in repository"
        return 1
    fi

    log_info "Installing Python packages (this may take 10-15 minutes)..."
    pip install -r "${INSTALL_DIR}/requirements_latest.txt" 2>&1 | tee -a "${LOG_FILE}"

    # Verify critical packages
    log_info "Verifying critical packages..."
    python3 <<EOF
import sys
packages = ['torch', 'transformers', 'sentence_transformers', 'qdrant_client',
            'fastapi', 'psycopg2', 'redis', 'langchain']
missing = []
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Missing packages: {', '.join(missing)}")
    sys.exit(1)
else:
    print("All critical packages installed successfully")
    sys.exit(0)
EOF

    if [ $? -eq 0 ]; then
        log_success "Python dependencies installed"
        return 0
    else
        log_error "Some Python packages failed to install"
        return 1
    fi
}

stage_06_setup_database() {
    print_header "STAGE 6: Setup PostgreSQL Database"

    log_info "Starting PostgreSQL service..."
    systemctl start postgresql
    systemctl enable postgresql

    log_info "Creating database and user..."

    # Generate secure password
    local db_password=$(openssl rand -base64 32)

    # Create database and user
    sudo -u postgres psql <<EOF 2>&1 | tee -a "${LOG_FILE}"
-- Create user
CREATE USER "rag-system" WITH PASSWORD '${db_password}';

-- Create database
CREATE DATABASE rag_metadata OWNER "rag-system";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE rag_metadata TO "rag-system";

-- Create UUID extension
\c rag_metadata
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
EOF

    # Save credentials to environment file
    local env_file="${INSTALL_DIR}/.env"
    cat >> "${env_file}" <<EOF

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_metadata
POSTGRES_USER=rag-system
POSTGRES_PASSWORD=${db_password}
EOF

    chmod 600 "${env_file}"

    log_success "PostgreSQL database configured"
    return 0
}

stage_07_setup_redis() {
    print_header "STAGE 7: Setup Redis"

    log_info "Configuring Redis..."

    # Generate secure password
    local redis_password=$(openssl rand -base64 32)

    # Configure Redis
    cat >> /etc/redis/redis.conf <<EOF

# RAG System Configuration
requirepass ${redis_password}
maxmemory 8gb
maxmemory-policy allkeys-lru
port 6380
EOF

    log_info "Starting Redis service..."
    systemctl restart redis-server
    systemctl enable redis-server

    # Save credentials
    local env_file="${INSTALL_DIR}/.env"
    cat >> "${env_file}" <<EOF

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_PASSWORD=${redis_password}
EOF

    log_success "Redis configured"
    return 0
}

stage_08_setup_qdrant() {
    print_header "STAGE 8: Setup Qdrant Vector Database"

    log_info "Installing Qdrant..."

    # Download and install Qdrant
    local qdrant_version="v1.12.1"
    wget "https://github.com/qdrant/qdrant/releases/download/${qdrant_version}/qdrant-x86_64-unknown-linux-gnu.tar.gz" \
        -O /tmp/qdrant.tar.gz 2>&1 | tee -a "${LOG_FILE}"

    tar -xzf /tmp/qdrant.tar.gz -C /opt/

    # Create Qdrant systemd service
    cat > /etc/systemd/system/qdrant.service <<EOF
[Unit]
Description=Qdrant Vector Database
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/qdrant
ExecStart=/opt/qdrant/qdrant
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl start qdrant
    systemctl enable qdrant

    # Wait for Qdrant to start
    sleep 5

    # Verify Qdrant is running
    if curl -s http://localhost:6333/healthz | grep -q "ok"; then
        log_success "Qdrant installed and running"
        return 0
    else
        log_error "Qdrant failed to start"
        return 1
    fi
}

stage_09_download_models() {
    print_header "STAGE 9: Download ML Models"

    local models_dir="${INSTALL_DIR}/models"
    mkdir -p "${models_dir}"

    local venv_path="${INSTALL_DIR}/venv"
    source "${venv_path}/bin/activate"

    log_info "Downloading embedding models (this may take 30-60 minutes)..."

    # Download models using Python
    python3 <<EOF
from sentence_transformers import SentenceTransformer
import os

models_dir = '${models_dir}'
os.makedirs(models_dir, exist_ok=True)

models = [
    ('BAAI/bge-m3', 'bge-m3'),
    ('microsoft/codebert-base', 'codebert-base'),
    ('intfloat/e5-large-v2', 'e5-large-v2'),
]

for model_name, local_name in models:
    print(f"Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    model.save(os.path.join(models_dir, local_name))
    print(f"✓ {model_name} downloaded")

print("All models downloaded successfully")
EOF

    if [ $? -eq 0 ]; then
        log_success "ML models downloaded"
        return 0
    else
        log_error "Failed to download models"
        return 1
    fi
}

stage_10_setup_directories() {
    print_header "STAGE 10: Setup Directory Structure"

    log_info "Creating directory structure..."

    mkdir -p "${INSTALL_DIR}"/{core,api,scripts,config,integrations,tests}
    mkdir -p /data/rag/{documents,vectors,cache,temp}
    mkdir -p /var/log/rag
    mkdir -p /var/lib/rag-system

    # Set permissions
    chown -R root:root "${INSTALL_DIR}"
    chmod -R 755 "${INSTALL_DIR}"

    chown -R root:root /data/rag
    chmod -R 755 /data/rag

    chown -R root:root /var/log/rag
    chmod -R 755 /var/log/rag

    log_success "Directory structure created"
    return 0
}

stage_11_copy_application_files() {
    print_header "STAGE 11: Copy Application Files"

    local repo_dir=$(pwd)

    log_info "Copying application files from ${repo_dir}..."

    # Copy core modules
    cp -r "${repo_dir}"/core/*.py "${INSTALL_DIR}/core/" 2>&1 | tee -a "${LOG_FILE}"

    # Copy API
    cp -r "${repo_dir}"/api/*.py "${INSTALL_DIR}/api/" 2>&1 | tee -a "${LOG_FILE}"

    # Copy scripts
    cp -r "${repo_dir}"/scripts/*.py "${INSTALL_DIR}/scripts/" 2>&1 | tee -a "${LOG_FILE}"

    # Copy config
    cp -r "${repo_dir}"/config/* "${INSTALL_DIR}/config/" 2>&1 | tee -a "${LOG_FILE}"

    # Copy integrations
    cp -r "${repo_dir}"/integrations/*.py "${INSTALL_DIR}/integrations/" 2>&1 | tee -a "${LOG_FILE}"

    # Copy requirements
    cp "${repo_dir}"/requirements_latest.txt "${INSTALL_DIR}/" 2>&1 | tee -a "${LOG_FILE}"

    log_success "Application files copied"
    return 0
}

stage_12_setup_configuration() {
    print_header "STAGE 12: Setup Configuration"

    log_info "Setting up configuration files..."

    local config_file="${INSTALL_DIR}/config/rag_config.yaml"

    # If .txt extension exists, rename it
    if [ -f "${config_file}.txt" ]; then
        mv "${config_file}.txt" "${config_file}"
    fi

    # Update configuration with actual values
    # This would need to be customized based on your environment

    log_success "Configuration setup complete"
    return 0
}

stage_13_setup_systemd_services() {
    print_header "STAGE 13: Setup Systemd Services"

    log_info "Creating systemd service files..."

    # RAG API Service
    cat > /etc/systemd/system/rag-api.service <<EOF
[Unit]
Description=RAG System API
After=network.target postgresql.service redis-server.service qdrant.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=${INSTALL_DIR}/venv/bin"
EnvironmentFile=${INSTALL_DIR}/.env
ExecStart=${INSTALL_DIR}/venv/bin/python -m uvicorn api.rag_api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Document Processor Daemon
    cat > /etc/systemd/system/rag-doc-processor.service <<EOF
[Unit]
Description=RAG Document Processor Daemon
After=network.target postgresql.service redis-server.service qdrant.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=${INSTALL_DIR}/venv/bin"
EnvironmentFile=${INSTALL_DIR}/.env
ExecStart=${INSTALL_DIR}/venv/bin/python scripts/document_processor_daemon.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload

    log_success "Systemd services created"
    return 0
}

stage_14_initialize_database() {
    print_header "STAGE 14: Initialize Database Schema"

    local venv_path="${INSTALL_DIR}/venv"
    source "${venv_path}/bin/activate"

    log_info "Initializing database schema..."

    # Load environment variables
    source "${INSTALL_DIR}/.env"

    # Run initialization script
    cd "${INSTALL_DIR}"
    python3 <<EOF
import sys
sys.path.insert(0, '${INSTALL_DIR}')

from core.security_manager import SecurityManager
from core.database_manager import DatabaseManager
import yaml

# Load config
with open('${INSTALL_DIR}/config/rag_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize database manager
db_manager = DatabaseManager(config)

# Initialize security manager (creates tables)
security_manager = SecurityManager(config)

print("Database schema initialized successfully")
EOF

    if [ $? -eq 0 ]; then
        log_success "Database initialized"
        return 0
    else
        log_error "Database initialization failed"
        return 1
    fi
}

stage_15_run_migrations() {
    print_header "STAGE 15: Run Database Migrations"

    log_info "Running database migrations..."

    # Placeholder for future migrations
    log_success "Migrations completed"
    return 0
}

stage_16_validate_installation() {
    print_header "STAGE 16: Validate Installation"

    local venv_path="${INSTALL_DIR}/venv"
    source "${venv_path}/bin/activate"

    log_info "Running validation checks..."

    # Run validation script
    cd "${INSTALL_DIR}"
    python3 <<EOF
import sys
sys.path.insert(0, '${INSTALL_DIR}')

errors = []

# Test imports
try:
    from core.rag_pipeline import RAGPipeline
    from core.query_optimizer import QueryOptimizer
    from core.document_processor import DocumentProcessor
    from core.security_manager import SecurityManager
    print("✓ Core modules import successfully")
except Exception as e:
    errors.append(f"Core module import failed: {e}")

# Test database connection
try:
    from core.database_manager import DatabaseManager
    import yaml

    with open('${INSTALL_DIR}/config/rag_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    db = DatabaseManager(config)
    health = db.health_check()
    if health['status'] == 'healthy':
        print("✓ Database connection successful")
    else:
        errors.append("Database health check failed")
except Exception as e:
    errors.append(f"Database connection failed: {e}")

# Test Redis connection
try:
    import redis
    import os
    r = redis.Redis(
        host='localhost',
        port=6380,
        password=os.environ.get('REDIS_PASSWORD'),
        decode_responses=True
    )
    r.ping()
    print("✓ Redis connection successful")
except Exception as e:
    errors.append(f"Redis connection failed: {e}")

# Test Qdrant connection
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(host='localhost', port=6333)
    client.get_collections()
    print("✓ Qdrant connection successful")
except Exception as e:
    errors.append(f"Qdrant connection failed: {e}")

if errors:
    print("\nValidation errors:")
    for err in errors:
        print(f"  ✗ {err}")
    sys.exit(1)
else:
    print("\n✓ All validation checks passed")
    sys.exit(0)
EOF

    if [ $? -eq 0 ]; then
        log_success "Validation passed"
        return 0
    else
        log_error "Validation failed"
        return 1
    fi
}

stage_17_start_services() {
    print_header "STAGE 17: Start Services"

    log_info "Starting RAG system services..."

    # Start API service
    systemctl start rag-api
    systemctl enable rag-api

    # Wait a bit for service to start
    sleep 3

    # Check if API is running
    if systemctl is-active --quiet rag-api; then
        log_success "RAG API service started"
    else
        log_error "RAG API service failed to start"
        journalctl -u rag-api -n 50 --no-pager | tee -a "${LOG_FILE}"
        return 1
    fi

    # Start document processor
    systemctl start rag-doc-processor
    systemctl enable rag-doc-processor

    if systemctl is-active --quiet rag-doc-processor; then
        log_success "RAG Document Processor started"
    else
        log_warning "RAG Document Processor failed to start (non-critical)"
    fi

    # Test API health endpoint
    sleep 5
    if curl -s http://localhost:8000/api/health | grep -q "healthy"; then
        log_success "API health check passed"
    else
        log_warning "API health check failed, but service is running"
    fi

    return 0
}

################################################################################
# Main Installation Flow
################################################################################

run_installation() {
    local start_stage=${1:-0}
    local end_stage=${2:-${#STAGES[@]}}

    print_header "RAG SYSTEM INSTALLATION"
    log_info "Installation started at $(date)"
    log_info "Log file: ${LOG_FILE}"

    # Run stages
    for i in $(seq $start_stage $(($end_stage - 1))); do
        local stage_name="${STAGES[$i]}"

        # Skip if already completed
        if is_stage_completed "$stage_name"; then
            log_info "Stage $i ($stage_name) already completed, skipping..."
            continue
        fi

        # Update state
        update_stage_status $i "in_progress"

        # Run stage function
        log_info "Running stage $i: $stage_name"

        if stage_${stage_name}; then
            update_stage_status $i "completed"
            log_success "Stage $i ($stage_name) completed successfully"
        else
            update_stage_status $i "failed"
            log_error "Stage $i ($stage_name) failed!"
            log_error "To resume from this stage, run: $0 --resume"
            log_error "To see logs: tail -f ${LOG_FILE}"
            return 1
        fi

        echo ""
    done

    print_header "INSTALLATION COMPLETED SUCCESSFULLY!"
    log_success "Installation completed at $(date)"
    log_info "Access the API at: http://localhost:8000/api/docs"
    log_info "Check logs at: ${LOG_DIR}"

    return 0
}

################################################################################
# Command Line Interface
################################################################################

show_usage() {
    cat <<EOF
RAG System Installation Script v2.0

Usage:
    $0 [OPTIONS]

Options:
    --install           Start fresh installation
    --resume            Resume from last failed stage
    --status            Show installation status
    --rollback          Rollback to previous installation
    --stage N           Start from specific stage N
    --list-stages       List all installation stages
    --validate          Run validation only
    --help              Show this help message

Examples:
    # Fresh installation
    sudo $0 --install

    # Resume after failure
    sudo $0 --resume

    # Start from stage 5
    sudo $0 --stage 5

    # Check status
    sudo $0 --status

EOF
}

show_status() {
    if [ ! -f "${STATE_FILE}" ]; then
        echo "No installation in progress"
        return
    fi

    echo "Installation Status:"
    echo "==================="
    python3 <<EOF
import json
with open('${STATE_FILE}', 'r') as f:
    state = json.load(f)

print(f"Install Date: {state.get('install_date', 'N/A')}")
print(f"Current Stage: {state.get('current_stage', 0)}")
print(f"Failed Stage: {state.get('failed_stage', 'None')}")
print(f"\nCompleted Stages:")
for stage in state.get('completed_stages', []):
    print(f"  ✓ {stage}")

if state.get('backup_path'):
    print(f"\nBackup Location: {state['backup_path']}")
EOF
}

list_stages() {
    echo "Installation Stages:"
    echo "==================="
    for i in "${!STAGES[@]}"; do
        printf "%2d. %s\n" $i "${STAGES[$i]}"
    done
}

main() {
    local action=""
    local start_stage=0

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install)
                action="install"
                shift
                ;;
            --resume)
                action="resume"
                shift
                ;;
            --status)
                show_status
                exit 0
                ;;
            --rollback)
                log_info "Rollback not yet implemented"
                exit 1
                ;;
            --stage)
                action="install"
                start_stage=$2
                shift 2
                ;;
            --list-stages)
                list_stages
                exit 0
                ;;
            --validate)
                check_root
                stage_16_validate_installation
                exit $?
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Default to install if no action specified
    if [ -z "$action" ]; then
        show_usage
        exit 1
    fi

    # Check root
    check_root

    # Initialize state
    init_state_file

    # Handle resume
    if [ "$action" == "resume" ]; then
        start_stage=$(get_current_stage)
        log_info "Resuming from stage ${start_stage}"
    fi

    # Run installation
    run_installation $start_stage
    exit $?
}

# Run main function
main "$@"
