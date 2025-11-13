#!/bin/bash
# Complete RAG System Deployment Script
# Deploys the full RAG system with all components

set -e

# Configuration
RAG_HOME="/opt/rag-system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER="rag-system"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
    
    # Check if base installation was completed
    if [[ ! -f "$RAG_HOME/INSTALLATION_SUMMARY.md" ]]; then
        error "Base installation not found. Please run complete_rag_installation.sh first"
    fi
    
    # Check if services are running
    services=("postgresql" "redis-rag" "qdrant")
    for service in "${services[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            warn "Service $service is not running. Starting..."
            systemctl start "$service"
        fi
    done
    
    log "Prerequisites check completed"
}

# Deploy RAG application code
deploy_application() {
    log "Deploying RAG application code..."
    
    # Create application directory structure
    mkdir -p "$RAG_HOME"/{core,api,config,templates,scripts,logs}
    
    # Copy application files (assuming they're in the current directory)
    if [[ -d "rag_system" ]]; then
        cp -r rag_system/* "$RAG_HOME/"
        chown -R "$USER:$USER" "$RAG_HOME"
    else
        error "RAG system source code not found in current directory"
    fi
    
    # Copy configuration files
    if [[ -f "rag_config.yaml" ]]; then
        cp rag_config.yaml "$RAG_HOME/config/"
    else
        warn "Configuration file not found, using default"
    fi
    
    log "Application code deployed"
}

# Setup Python application
setup_python_app() {
    log "Setting up Python application..."
    
    # Install additional Python packages for RAG system
    su - "$USER" -c "
        source $RAG_HOME/venv/bin/activate
        
        # Install RAG-specific packages
        pip install --no-cache-dir \
            fastapi==0.104.1 \
            uvicorn[standard]==0.24.0 \
            websockets==12.0 \
            streamlit==1.28.2 \
            gradio==4.8.0 \
            langchain==0.1.0 \
            qdrant-client==1.7.0 \
            sentence-transformers==2.2.2 \
            unstructured[local-inference]==0.11.8 \
            tree-sitter==0.20.4 \
            tree-sitter-c==0.20.6 \
            tree-sitter-cpp==0.20.3 \
            tree-sitter-python==0.20.4 \
            PyMuPDF==1.23.14 \