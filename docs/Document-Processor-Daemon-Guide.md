# Document Processor Daemon - Operations Guide

## Overview

The Document Processor Daemon is an automatic document ingestion service that monitors directories for new files and processes them into the RAG system automatically.

---

## How It Starts

### 1. **During Installation** (Automatic)

The daemon is automatically configured and started during the installation process:

**Stage 13** - Creates systemd service:
```bash
# Creates: /etc/systemd/system/rag-doc-processor.service
sudo ./install_rag_system.sh --install
```

**Stage 17** - Starts the service:
```bash
# Service is started and enabled
systemctl start rag-doc-processor
systemctl enable rag-doc-processor
```

### 2. **After Installation** (Manual)

If the service is stopped, you can start it manually:

```bash
# Start the service
sudo systemctl start rag-doc-processor

# Enable to start on boot
sudo systemctl enable rag-doc-processor

# Check status
sudo systemctl status rag-doc-processor
```

---

## Systemd Service Configuration

The service is defined at: `/etc/systemd/system/rag-doc-processor.service`

```ini
[Unit]
Description=RAG Document Processor Daemon
After=network.target postgresql.service redis-server.service qdrant.service
Requires=postgresql.service redis-server.service qdrant.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/rag-system
Environment="PATH=/opt/rag-system/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=/opt/rag-system"
EnvironmentFile=/opt/rag-system/.env
ExecStart=/opt/rag-system/venv/bin/python scripts/document_processor_daemon.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
TimeoutStartSec=300
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

---

## Service Management Commands

### Basic Operations

```bash
# Start the daemon
sudo systemctl start rag-doc-processor

# Stop the daemon
sudo systemctl stop rag-doc-processor

# Restart the daemon
sudo systemctl restart rag-doc-processor

# Check status
sudo systemctl status rag-doc-processor

# View logs in real-time
sudo journalctl -u rag-doc-processor -f

# View recent logs
sudo journalctl -u rag-doc-processor -n 100

# Enable auto-start on boot
sudo systemctl enable rag-doc-processor

# Disable auto-start
sudo systemctl disable rag-doc-processor
```

### Advanced Operations

```bash
# Reload systemd configuration (after editing service file)
sudo systemctl daemon-reload

# Show service properties
systemctl show rag-doc-processor

# Check if service is active
systemctl is-active rag-doc-processor

# Check if service is enabled
systemctl is-enabled rag-doc-processor

# Get service PID
systemctl show -p MainPID rag-doc-processor
```

---

## What The Daemon Does

### 1. **Monitors Directories**

Watches these directories for new files:
```
/data/rag/incoming/drivers/     # Driver source code and docs
/data/rag/incoming/embedded/    # Embedded/firmware code and docs
/data/rag/incoming/radar/       # Radar-specific documents
/data/rag/incoming/general/     # General documents
```

### 2. **Automatic Processing**

When a new file is detected:
1. File is validated (format, size)
2. Security classification is determined
3. Document is processed and chunked
4. Embeddings are generated
5. Vectors are stored in Qdrant
6. Metadata is stored in PostgreSQL
7. Original file is moved to processed directory

### 3. **Scheduled Tasks**

- **Hourly**: Cleanup of processed files
- **Daily**: Generate processing report

---

## Directory Structure

```
/data/rag/
├── incoming/           # Drop files here for processing
│   ├── drivers/       # Driver-related documents
│   ├── embedded/      # Embedded/firmware documents
│   ├── radar/         # Radar-specific documents
│   └── general/       # General documents
├── processed/         # Successfully processed files
│   └── YYYY-MM-DD/    # Organized by date
├── failed/            # Files that failed processing
│   └── YYYY-MM-DD/    # Organized by date
└── quarantine/        # Files with security issues
```

---

## How to Use

### Method 1: Copy Files to Watched Directory

```bash
# For driver documentation
sudo cp my-driver-doc.pdf /data/rag/incoming/drivers/

# For embedded code
sudo cp firmware.c /data/rag/incoming/embedded/

# For radar documents
sudo cp radar-spec.docx /data/rag/incoming/radar/

# For general documents
sudo cp general-doc.pdf /data/rag/incoming/general/
```

The daemon will automatically detect and process the files.

### Method 2: Batch Copy

```bash
# Copy entire directory
sudo cp -r /path/to/driver/docs/* /data/rag/incoming/drivers/

# The daemon processes files one by one
```

### Method 3: Rsync from Remote Server

```bash
# Sync from remote server
rsync -av --progress remote-server:/docs/ /data/rag/incoming/general/
```

---

## Monitoring

### 1. **Check Service Status**

```bash
sudo systemctl status rag-doc-processor
```

**Healthy output**:
```
● rag-doc-processor.service - RAG Document Processor Daemon
     Loaded: loaded (/etc/systemd/system/rag-doc-processor.service; enabled)
     Active: active (running) since Mon 2025-01-13 10:00:00 UTC; 2h ago
   Main PID: 12345 (python)
      Tasks: 5 (limit: 38327)
     Memory: 1.2G
```

### 2. **View Logs**

```bash
# Real-time logs
sudo journalctl -u rag-doc-processor -f

# Last 100 lines
sudo journalctl -u rag-doc-processor -n 100

# Logs from today
sudo journalctl -u rag-doc-processor --since today

# Logs from last hour
sudo journalctl -u rag-doc-processor --since "1 hour ago"

# Application logs
tail -f /var/log/rag/document_processor.log
```

### 3. **Check Statistics**

The daemon logs statistics periodically:

```bash
grep "Statistics" /var/log/rag/document_processor.log
```

Output:
```
2025-01-13 10:30:00 - Statistics: processed=45, failed=2, queue_size=3
```

### 4. **Monitor Incoming Directory**

```bash
# Check files waiting to be processed
ls -lh /data/rag/incoming/*/

# Count files in queue
find /data/rag/incoming -type f | wc -l

# Check disk usage
du -sh /data/rag/incoming/*
```

---

## Troubleshooting

### Issue 1: Service Won't Start

**Check status**:
```bash
sudo systemctl status rag-doc-processor
```

**Common causes**:

1. **Missing dependencies**:
```bash
# Check if RAG pipeline can be imported
sudo -u root /opt/rag-system/venv/bin/python -c "from core.rag_pipeline import RAGPipeline"
```

2. **Database not running**:
```bash
sudo systemctl status postgresql
sudo systemctl status redis-server
sudo systemctl status qdrant
```

3. **Permission issues**:
```bash
# Check directory permissions
sudo ls -ld /data/rag/incoming
sudo ls -ld /var/log/rag
```

**Fix**:
```bash
# Restart dependencies first
sudo systemctl restart postgresql redis-server qdrant

# Then restart daemon
sudo systemctl restart rag-doc-processor
```

### Issue 2: Files Not Being Processed

**Check**:
```bash
# 1. Is service running?
sudo systemctl status rag-doc-processor

# 2. Are files in the right directory?
ls -lh /data/rag/incoming/*/

# 3. Check logs for errors
sudo journalctl -u rag-doc-processor -n 50
```

**Common causes**:
- Files in wrong directory
- Unsupported file format
- File permissions prevent reading
- Service crashed

**Fix**:
```bash
# Ensure files are readable
sudo chmod 644 /data/rag/incoming/*/*.pdf

# Restart service
sudo systemctl restart rag-doc-processor
```

### Issue 3: High Memory Usage

**Check memory**:
```bash
# Service memory usage
systemctl status rag-doc-processor | grep Memory

# System memory
free -h

# Top processes
top -u root
```

**Fix** (if memory is too high):

1. **Adjust batch size** in config:
```bash
# Edit: /opt/rag-system/config/rag_config.yaml
document_processing:
  max_batch_size: 10  # Reduce from 50
  processing_timeout_seconds: 600
```

2. **Restart service**:
```bash
sudo systemctl restart rag-doc-processor
```

### Issue 4: Files Going to Failed Directory

**Check failed files**:
```bash
ls -lh /data/rag/failed/$(date +%Y-%m-%d)/
```

**Check why they failed**:
```bash
grep "ERROR" /var/log/rag/document_processor.log | tail -20
```

**Common reasons**:
- Corrupted file
- Unsupported format
- File too large
- Encoding issues

**Fix**:
```bash
# Move back to incoming to retry
sudo mv /data/rag/failed/2025-01-13/file.pdf /data/rag/incoming/general/
```

### Issue 5: Queue Building Up

**Check queue size**:
```bash
find /data/rag/incoming -type f | wc -l
```

**If queue is large** (>100 files):

1. **Check processing speed**:
```bash
# Monitor for 1 minute
watch -n 5 'find /data/rag/incoming -type f | wc -l'
```

2. **Increase parallelism** (edit daemon config if needed)

3. **Check for stuck files**:
```bash
# Files older than 1 hour in incoming
find /data/rag/incoming -type f -mmin +60
```

---

## Configuration

### Main Configuration File

Location: `/opt/rag-system/config/rag_config.yaml`

```yaml
document_processing:
  # Watch directories
  watch_directories:
    - /data/rag/incoming/drivers
    - /data/rag/incoming/embedded
    - /data/rag/incoming/radar
    - /data/rag/incoming/general

  # Supported formats
  supported_formats:
    - ".pdf"
    - ".docx"
    - ".doc"
    - ".txt"
    - ".md"
    - ".c"
    - ".h"
    - ".cpp"
    - ".py"
    # ... etc

  # Processing limits
  max_file_size_mb: 100
  max_batch_size: 50
  processing_timeout_seconds: 300

  # Daemon settings
  daemon:
    check_interval_seconds: 10
    cleanup_interval_hours: 1
    report_interval_hours: 24
```

### Environment Variables

Location: `/opt/rag-system/.env`

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

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/rag/document_processor.log
```

### Restart After Configuration Changes

```bash
# Reload configuration
sudo systemctl restart rag-doc-processor

# Or reload daemon and restart
sudo systemctl daemon-reload
sudo systemctl restart rag-doc-processor
```

---

## Manual Testing

### Test Single File Processing

```bash
# Activate virtual environment
cd /opt/rag-system
source venv/bin/activate

# Test processing a single file
python <<EOF
import asyncio
from core.rag_pipeline import RAGPipeline

async def test():
    pipeline = RAGPipeline()
    result = await pipeline.ingest_document(
        file_path="/path/to/test/file.pdf",
        security_classification="internal",
        domain="general"
    )
    print(f"Success: {result}")

asyncio.run(test())
EOF
```

### Test Directory Watch

```bash
# 1. Stop daemon
sudo systemctl stop rag-doc-processor

# 2. Run manually in foreground
cd /opt/rag-system
source venv/bin/activate
python scripts/document_processor_daemon.py

# 3. In another terminal, copy a file
sudo cp test.pdf /data/rag/incoming/general/

# 4. Watch the output in first terminal
# 5. Stop with Ctrl+C when done

# 6. Restart daemon
sudo systemctl start rag-doc-processor
```

---

## Performance Tuning

### For High Volume (1000+ docs/day)

```yaml
# In rag_config.yaml
document_processing:
  max_batch_size: 100
  processing_timeout_seconds: 600
  daemon:
    check_interval_seconds: 5
    parallel_workers: 4
```

### For Low Volume (< 100 docs/day)

```yaml
document_processing:
  max_batch_size: 20
  processing_timeout_seconds: 300
  daemon:
    check_interval_seconds: 30
    parallel_workers: 2
```

---

## Integration with Other Systems

### 1. **Automated File Drop via Cron**

```bash
# Add to crontab: sudo crontab -e
# Sync from network share every hour
0 * * * * rsync -av /mnt/network-docs/ /data/rag/incoming/general/
```

### 2. **Post-Build Hook (CI/CD)**

```bash
# In your CI/CD pipeline
after_build:
  - scp documentation/*.pdf rag-server:/data/rag/incoming/drivers/
  - scp src/**/*.h rag-server:/data/rag/incoming/drivers/
```

### 3. **Email Attachments**

Use a mail processor to extract attachments:
```bash
# Simple example with procmail
:0
* ^Subject:.*Documentation
| python /opt/rag-system/scripts/process_email_attachments.py
```

### 4. **Web Upload Interface**

API endpoint for file upload:
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf" \
  -F "domain=drivers" \
  -F "security_level=internal"
```

---

## Best Practices

1. **Organize by Domain**: Place files in appropriate domain directories
2. **Batch Upload**: For large imports, copy in batches to avoid overwhelming
3. **Monitor Regularly**: Check logs and queue size daily
4. **Clean Up**: Remove very old files from processed directory
5. **Backup**: Backup incoming directory before processing large batches
6. **Test Files**: Test with sample files before bulk uploads
7. **Resource Monitoring**: Watch CPU, memory, and disk during peak loads

---

## Emergency Procedures

### Stop Processing Immediately

```bash
# Stop daemon
sudo systemctl stop rag-doc-processor

# Stop all processing
sudo pkill -f document_processor_daemon
```

### Clear Queue

```bash
# Move all pending files out
sudo mv /data/rag/incoming/*/* /data/rag/incoming-backup/

# Restart daemon
sudo systemctl start rag-doc-processor
```

### Reset Daemon

```bash
# Stop service
sudo systemctl stop rag-doc-processor

# Clear logs
sudo truncate -s 0 /var/log/rag/document_processor.log

# Start fresh
sudo systemctl start rag-doc-processor
```

---

## FAQ

**Q: How long does it take to process a document?**
A: Typically 2-10 seconds per document, depending on size and complexity.

**Q: Can I process multiple directories simultaneously?**
A: Yes, the daemon watches all configured directories in parallel.

**Q: What happens if the daemon crashes during processing?**
A: The service automatically restarts (RestartSec=10). In-progress files will be reprocessed.

**Q: Can I pause processing?**
A: Yes, stop the service: `sudo systemctl stop rag-doc-processor`

**Q: How do I add a new watch directory?**
A: Edit `/opt/rag-system/config/rag_config.yaml`, add directory, restart service.

**Q: Can I run multiple daemons?**
A: Not recommended. One daemon can handle multiple directories efficiently.

**Q: What file formats are supported?**
A: PDF, DOCX, TXT, MD, C/C++, Python, and 15+ others. See config for full list.

**Q: How do I bulk import 10,000 documents?**
A: Copy in batches of 100-500, monitor queue size, ensure adequate disk space.

---

**Document Version**: 1.0
**Last Updated**: 2025-01-13
**Service**: rag-doc-processor
