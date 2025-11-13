# RAG System - Complete Installation Guide

## Version 2.0 - Stage-wise Installation with Resume Capability

This guide provides comprehensive instructions for installing the RAG system with support for resuming from failed stages.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Installation Stages](#installation-stages)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)
6. [Rollback](#rollback)
7. [Post-Installation](#post-installation)

---

## Prerequisites

### Hardware Requirements

**Minimum**:
- CPU: 16 cores
- RAM: 32GB
- Storage: 100GB free
- GPU: Optional (1x NVIDIA GPU with 16GB+ VRAM)

**Recommended**:
- CPU: 32+ cores
- RAM: 128GB
- Storage: 1TB SSD + 10TB HDD
- GPU: 4x NVIDIA L40 (24GB each) or equivalent

### Software Requirements

- **OS**: Ubuntu 22.04 LTS (recommended) or Ubuntu 20.04 LTS
- **Python**: 3.11 or higher
- **CUDA**: 12.1+ (if using GPU)
- **Docker**: Optional, for containerized deployment

### Network Requirements

- Internet connection for downloading packages and models (first-time setup)
- For air-gapped deployment, prepare offline package repository

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourorg/rag-install.git
cd rag-install
```

### 2. Make Scripts Executable

```bash
chmod +x install_rag_system.sh
chmod +x rollback_installation.sh
```

### 3. Run Installation

```bash
sudo ./install_rag_system.sh --install
```

That's it! The script will guide you through all stages.

---

## Installation Stages

The installation is divided into 17 stages. If any stage fails, you can resume from that point.

### Stage Overview

| Stage | Name | Description | Duration |
|-------|------|-------------|----------|
| 01 | Preflight Checks | Verify system requirements | 1-2 min |
| 02 | Backup Existing | Backup current installation | 2-5 min |
| 03 | Install System Deps | Install OS packages | 5-10 min |
| 04 | Setup Python Env | Create virtual environment | 2-3 min |
| 05 | Install Python Deps | Install Python packages | 10-15 min |
| 06 | Setup Database | Configure PostgreSQL | 2-3 min |
| 07 | Setup Redis | Configure Redis cache | 1-2 min |
| 08 | Setup Qdrant | Install vector database | 2-3 min |
| 09 | Download Models | Download ML models | 30-60 min |
| 10 | Setup Directories | Create directory structure | 1 min |
| 11 | Copy Application Files | Copy code files | 1-2 min |
| 12 | Setup Configuration | Configure system | 1-2 min |
| 13 | Setup Systemd Services | Create services | 1 min |
| 14 | Initialize Database | Create DB schema | 2-3 min |
| 15 | Run Migrations | Apply DB migrations | 1 min |
| 16 | Validate Installation | Run validation checks | 2-3 min |
| 17 | Start Services | Start all services | 2-3 min |

**Total Time**: 60-90 minutes (depending on internet speed for model downloads)

### Detailed Stage Information

#### Stage 01: Preflight Checks

Validates:
- Operating system compatibility
- Disk space (minimum 50GB)
- RAM (minimum 32GB)
- Python version (3.11+)
- GPU availability (optional)
- Network connectivity

**If this stage fails**: Fix the reported issue and re-run.

#### Stage 09: Download Models

Downloads three embedding models:
- **BGE-M3**: Multi-lingual embeddings (~2GB)
- **CodeBERT**: Code-specific embeddings (~500MB)
- **E5-large-v2**: General embeddings (~1GB)

**Note**: This is the longest stage. If it fails due to network issues, simply resume:

```bash
sudo ./install_rag_system.sh --resume
```

---

## Advanced Usage

### Resume from Failed Stage

If installation fails at any stage:

```bash
# Check status
sudo ./install_rag_system.sh --status

# Resume from last failed stage
sudo ./install_rag_system.sh --resume
```

### Start from Specific Stage

If you want to skip early stages (e.g., you already have dependencies installed):

```bash
# Start from stage 9 (download models)
sudo ./install_rag_system.sh --stage 9
```

### List All Stages

```bash
sudo ./install_rag_system.sh --list-stages
```

### Check Installation Status

```bash
sudo ./install_rag_system.sh --status
```

Output example:
```
Installation Status:
===================
Install Date: 2025-01-13T10:30:00
Current Stage: 9
Failed Stage: None

Completed Stages:
  ✓ 01_preflight_checks
  ✓ 02_backup_existing
  ✓ 03_install_system_deps
  ✓ 04_setup_python_env
  ✓ 05_install_python_deps
  ✓ 06_setup_database
  ✓ 07_setup_redis
  ✓ 08_setup_qdrant

Backup Location: /opt/rag-system-backup/rag-system-backup-20250113_103000.tar.gz
```

### Validate Installation Only

After installation, you can re-run validation:

```bash
sudo ./install_rag_system.sh --validate
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Insufficient Disk Space

**Error**: "Insufficient disk space. Need at least 50GB"

**Solution**:
```bash
# Check available space
df -h /

# Clean up if needed
sudo apt-get clean
sudo apt-get autoremove

# Or mount additional storage
```

#### Issue 2: Python Version Too Old

**Error**: "Python 3.11+ required, found 3.8"

**Solution**:
```bash
# Add deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

# Install Python 3.11
sudo apt-get install python3.11 python3.11-venv python3.11-dev

# Set as default (optional)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

#### Issue 3: Model Download Fails

**Error**: "Failed to download models"

**Solution**:
```bash
# Check internet connectivity
ping -c 3 huggingface.co

# If behind proxy, set proxy
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# Resume installation
sudo -E ./install_rag_system.sh --resume
```

#### Issue 4: Database Connection Failed

**Error**: "Database connection failed"

**Solution**:
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
sudo -u postgres psql -l

# Restart PostgreSQL
sudo systemctl restart postgresql

# Resume installation
sudo ./install_rag_system.sh --resume
```

#### Issue 5: GPU Not Detected

**Warning**: "No NVIDIA GPUs detected"

**Solution** (if you have GPU):
```bash
# Install NVIDIA drivers
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# Install CUDA
# Follow: https://developer.nvidia.com/cuda-downloads

# Reboot and verify
nvidia-smi
```

**Note**: System will work on CPU (slower) if no GPU is available.

### Viewing Logs

Installation logs are saved in `/var/log/rag-install/`:

```bash
# View latest log
sudo tail -f /var/log/rag-install/install_*.log

# View specific stage logs
sudo grep "STAGE 09" /var/log/rag-install/install_*.log

# View errors only
sudo grep "ERROR" /var/log/rag-install/install_*.log
```

### Debug Mode

For more verbose output, modify the script:

```bash
# Edit install_rag_system.sh
# Add at the top:
set -x  # Enable debug mode
```

---

## Rollback

If you need to revert to a previous installation:

### List Available Backups

```bash
sudo ./rollback_installation.sh --list
```

### Automatic Rollback

Rollback to the backup created before current installation:

```bash
sudo ./rollback_installation.sh --auto
```

### Rollback to Specific Backup

```bash
sudo ./rollback_installation.sh --file /opt/rag-system-backup/rag-system-backup-20250113.tar.gz
```

**Note**: Rollback will:
1. Create a pre-rollback backup of current installation
2. Stop all services
3. Remove current installation
4. Restore from specified backup
5. Restart services

---

## Post-Installation

### 1. Verify Services

```bash
# Check all services
sudo systemctl status rag-api
sudo systemctl status rag-doc-processor
sudo systemctl status postgresql
sudo systemctl status redis-server
sudo systemctl status qdrant

# Check API health
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "database": "ok",
    "redis": "ok",
    "qdrant": "ok"
  }
}
```

### 2. Access API Documentation

Open in browser:
```
http://localhost:8000/api/docs
```

This provides interactive API documentation (Swagger UI).

### 3. Create Admin User

```bash
cd /opt/rag-system
source venv/bin/activate
python scripts/create_users.py
```

Follow the prompts to create admin user.

### 4. Test Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "How to initialize device driver?",
    "top_k": 5
  }'
```

### 5. Ingest Sample Documents

```bash
cd /opt/rag-system
source venv/bin/activate

python <<EOF
from core.rag_pipeline import RAGPipeline
import asyncio

async def main():
    pipeline = RAGPipeline()

    # Ingest a directory
    stats = await pipeline.batch_ingest(
        directory_path="/path/to/your/documents",
        recursive=True
    )
    print(f"Ingested {stats['successful']} documents")

asyncio.run(main())
EOF
```

### 6. Setup Bugzilla Integration (Optional)

If you want to sync bugs from Bugzilla:

```bash
# Set Bugzilla credentials
export BUGZILLA_API_KEY=your_api_key_here

# Edit config
vim /opt/rag-system/config/rag_config.yaml

# Update Bugzilla section:
integrations:
  bugzilla:
    enabled: true
    url: "https://your-bugzilla.example.com"
    sync_interval_hours: 24
    products: ['BSP', 'Drivers', 'Radar']

# Restart services
sudo systemctl restart rag-api
```

### 7. Monitor System

```bash
# View API logs
sudo journalctl -u rag-api -f

# View document processor logs
sudo journalctl -u rag-doc-processor -f

# Check system resources
htop
nvidia-smi  # For GPU monitoring
```

### 8. Setup Nginx (Production)

For production deployment with SSL:

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Configure Nginx
sudo vim /etc/nginx/sites-available/rag-system

# Add configuration (example below)

# Enable site
sudo ln -s /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

Example Nginx configuration:
```nginx
server {
    listen 80;
    server_name rag.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Maintenance

### Update System

To update the RAG system:

```bash
cd /home/user/rag-install
git pull origin main

# Run installation (will backup current version)
sudo ./install_rag_system.sh --install
```

### Backup Manually

```bash
# Create manual backup
sudo tar -czf /opt/rag-system-backup/manual-backup-$(date +%Y%m%d).tar.gz \
  -C /opt rag-system

# Backup database
sudo -u postgres pg_dump rag_metadata > /tmp/rag_db_backup.sql
```

### Clean Old Backups

```bash
# Keep only last 5 backups
cd /opt/rag-system-backup
ls -t *.tar.gz | tail -n +6 | xargs rm -f
```

### Monitor Disk Usage

```bash
# Check data directory
du -sh /data/rag/*

# Check logs
du -sh /var/log/rag/*

# Clean old logs (older than 30 days)
find /var/log/rag -name "*.log" -mtime +30 -delete
```

---

## Uninstallation

To completely remove the RAG system:

```bash
# Stop services
sudo systemctl stop rag-api rag-doc-processor

# Disable services
sudo systemctl disable rag-api rag-doc-processor

# Remove services
sudo rm /etc/systemd/system/rag-*.service
sudo systemctl daemon-reload

# Remove installation
sudo rm -rf /opt/rag-system

# Remove data (CAUTION: This deletes all documents and vectors!)
sudo rm -rf /data/rag

# Remove logs
sudo rm -rf /var/log/rag

# Remove state
sudo rm -rf /var/lib/rag-system

# Drop database
sudo -u postgres psql -c "DROP DATABASE rag_metadata;"
sudo -u postgres psql -c "DROP USER \"rag-system\";"

# Stop and remove Qdrant
sudo systemctl stop qdrant
sudo rm -rf /opt/qdrant
sudo rm /etc/systemd/system/qdrant.service
```

---

## Support

### Getting Help

1. **Check documentation**: Review this guide and other docs in `/docs/`
2. **View logs**: Check logs in `/var/log/rag-install/` and `/var/log/rag/`
3. **Run validation**: `sudo ./install_rag_system.sh --validate`
4. **Check status**: `sudo ./install_rag_system.sh --status`

### Reporting Issues

When reporting issues, include:
- Installation log file
- Output of `sudo ./install_rag_system.sh --status`
- System information: `uname -a`, `lsb_release -a`
- Error messages

### Contact

- Email: support@example.com
- Internal Wiki: https://wiki.example.com/rag-system
- Issue Tracker: https://github.com/yourorg/rag-install/issues

---

## Appendix

### A. System Requirements Details

#### CPU Requirements
- **Minimum**: 16 cores for basic operation
- **Recommended**: 32+ cores for production with multiple concurrent users
- Architecture: x86_64
- Features: AVX2 support recommended for optimized operations

#### Memory Requirements
- **32GB**: Basic operation, 5-10 concurrent users
- **64GB**: Medium load, 20-30 concurrent users
- **128GB**: Production load, 50+ concurrent users, optimal performance
- **256GB**: High load, 100+ concurrent users, extensive caching

#### Storage Requirements
- **NVMe SSD (500GB)**: For databases, indices, and hot data
- **SSD (2TB)**: For recent documents and model cache
- **HDD (10TB+)**: For archived documents and backups

### B. Port Requirements

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| RAG API | 8000 | HTTP | Main API endpoint |
| PostgreSQL | 5432 | TCP | Database |
| Redis | 6380 | TCP | Cache |
| Qdrant | 6333 | HTTP | Vector database |
| Qdrant gRPC | 6334 | gRPC | Vector database (optional) |

### C. Environment Variables

Key environment variables (stored in `/opt/rag-system/.env`):

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_metadata
POSTGRES_USER=rag-system
POSTGRES_PASSWORD=<generated>

# Redis
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_PASSWORD=<generated>

# Security
JWT_SECRET_KEY=<generated>
ENCRYPTION_KEY=<generated>

# Optional: Bugzilla
BUGZILLA_API_KEY=<your_key>
```

### D. File Locations

| Purpose | Location |
|---------|----------|
| Installation | `/opt/rag-system/` |
| Configuration | `/opt/rag-system/config/` |
| Models | `/opt/rag-system/models/` |
| Documents | `/data/rag/documents/` |
| Vectors | `/data/rag/vectors/` |
| Logs | `/var/log/rag/` |
| Install Logs | `/var/log/rag-install/` |
| State | `/var/lib/rag-system/` |
| Backups | `/opt/rag-system-backup/` |

---

**Document Version**: 1.0
**Last Updated**: 2025-01-13
**Maintained by**: RAG System Team
