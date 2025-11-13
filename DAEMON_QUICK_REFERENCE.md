# Document Processor Daemon - Quick Reference

## 🚀 Start/Stop Commands

```bash
# Start daemon
sudo systemctl start rag-doc-processor

# Stop daemon
sudo systemctl stop rag-doc-processor

# Restart daemon
sudo systemctl restart rag-doc-processor

# Check status
sudo systemctl status rag-doc-processor

# Enable auto-start on boot
sudo systemctl enable rag-doc-processor
```

---

## 📁 Drop Files Here

```bash
# For driver documents
/data/rag/incoming/drivers/

# For embedded/firmware
/data/rag/incoming/embedded/

# For radar documents
/data/rag/incoming/radar/

# For general documents
/data/rag/incoming/general/
```

**Usage**:
```bash
# Copy file to auto-process
sudo cp document.pdf /data/rag/incoming/drivers/

# The daemon will automatically:
# 1. Detect the file
# 2. Process and chunk it
# 3. Generate embeddings
# 4. Store in vector database
# 5. Move to /data/rag/processed/
```

---

## 📊 Monitoring

```bash
# View real-time logs
sudo journalctl -u rag-doc-processor -f

# Check last 50 log entries
sudo journalctl -u rag-doc-processor -n 50

# Application logs
tail -f /var/log/rag/document_processor.log

# Check queue size
find /data/rag/incoming -type f | wc -l

# Check processed today
ls /data/rag/processed/$(date +%Y-%m-%d)/ | wc -l

# Check failed files
ls /data/rag/failed/$(date +%Y-%m-%d)/ 2>/dev/null
```

---

## 🔍 Troubleshooting

### Daemon Not Running?
```bash
# Check status
sudo systemctl status rag-doc-processor

# View errors
sudo journalctl -u rag-doc-processor -n 50 | grep ERROR

# Restart dependencies
sudo systemctl restart postgresql redis-server qdrant

# Restart daemon
sudo systemctl restart rag-doc-processor
```

### Files Not Processing?
```bash
# 1. Check service is active
systemctl is-active rag-doc-processor

# 2. Check file permissions
ls -lh /data/rag/incoming/general/

# 3. Fix permissions if needed
sudo chmod 644 /data/rag/incoming/*/*.pdf

# 4. Check logs
sudo journalctl -u rag-doc-processor -n 20
```

### High Memory Usage?
```bash
# Check memory
systemctl status rag-doc-processor | grep Memory

# Restart to clear
sudo systemctl restart rag-doc-processor
```

### Queue Building Up?
```bash
# Check queue size
find /data/rag/incoming -type f | wc -l

# Find stuck files (>1 hour old)
find /data/rag/incoming -type f -mmin +60

# Clear queue (emergency)
sudo systemctl stop rag-doc-processor
sudo mv /data/rag/incoming/*/* /tmp/backup/
sudo systemctl start rag-doc-processor
```

---

## 📝 How It Works

```
┌─────────────────────────────────────────────────┐
│  1. File dropped in /data/rag/incoming/        │
│                                                 │
│  2. Daemon detects new file                    │
│                                                 │
│  3. Validate format & security                 │
│                                                 │
│  4. Process document:                          │
│     • Parse content                            │
│     • Chunk intelligently                      │
│     • Generate embeddings                      │
│                                                 │
│  5. Store:                                     │
│     • Vectors → Qdrant                         │
│     • Metadata → PostgreSQL                    │
│                                                 │
│  6. Move to /data/rag/processed/              │
│     (or /data/rag/failed/ if error)           │
└─────────────────────────────────────────────────┘
```

**Processing Time**: 2-10 seconds per document

---

## 🎯 Common Tasks

### Upload Single File
```bash
sudo cp my-document.pdf /data/rag/incoming/general/
```

### Bulk Upload
```bash
# Copy entire directory
sudo cp -r /path/to/docs/* /data/rag/incoming/drivers/

# Or use rsync
rsync -av --progress /path/to/docs/ /data/rag/incoming/drivers/
```

### Watch Processing in Real-Time
```bash
# Terminal 1: Watch logs
sudo journalctl -u rag-doc-processor -f

# Terminal 2: Copy files
sudo cp *.pdf /data/rag/incoming/general/
```

### Check Today's Statistics
```bash
# Processed today
find /data/rag/processed/$(date +%Y-%m-%d) -type f | wc -l

# Failed today
find /data/rag/failed/$(date +%Y-%m-%d) -type f 2>/dev/null | wc -l

# Still in queue
find /data/rag/incoming -type f | wc -l
```

### Manual Reprocess Failed File
```bash
# Move from failed back to incoming
sudo mv /data/rag/failed/2025-01-13/doc.pdf /data/rag/incoming/general/
```

---

## 📋 Supported File Formats

✅ **Documents**:
- PDF (.pdf)
- Word (.docx, .doc)
- Text (.txt, .md, .rst)
- OpenDocument (.odt)

✅ **Code**:
- C/C++ (.c, .h, .cpp, .hpp)
- Python (.py)
- MATLAB (.m)
- HTML (.html)
- SQL (.sql)

✅ **Data**:
- CSV (.csv)
- Excel (.xlsx)
- JSON (.json)

**Max File Size**: 100MB (configurable)

---

## ⚙️ Configuration

**Config File**: `/opt/rag-system/config/rag_config.yaml`

```yaml
document_processing:
  max_file_size_mb: 100
  max_batch_size: 50
  processing_timeout_seconds: 300

  daemon:
    check_interval_seconds: 10
    cleanup_interval_hours: 1
```

**After changing config**:
```bash
sudo systemctl restart rag-doc-processor
```

---

## 🔄 Service Lifecycle

### During Installation
```bash
# Stage 13: Service created
# Stage 17: Service started
sudo ./install_rag_system.sh --install
```

### After Reboot
```bash
# Auto-starts if enabled
systemctl is-enabled rag-doc-processor  # Check

# If not enabled
sudo systemctl enable rag-doc-processor
```

### Scheduled Tasks
- **Every 10 seconds**: Check for new files
- **Every hour**: Cleanup processed files (older than 30 days)
- **Every 24 hours**: Generate processing report

---

## 🚨 Emergency Procedures

### Stop Everything Immediately
```bash
sudo systemctl stop rag-doc-processor
```

### Clear All Pending Files
```bash
# Move to backup
sudo mkdir -p /tmp/incoming-backup
sudo mv /data/rag/incoming/*/* /tmp/incoming-backup/
```

### Reset and Start Fresh
```bash
# Stop service
sudo systemctl stop rag-doc-processor

# Clear logs
sudo truncate -s 0 /var/log/rag/document_processor.log

# Start
sudo systemctl start rag-doc-processor
```

### Force Kill (if stuck)
```bash
# Find PID
ps aux | grep document_processor_daemon

# Kill it
sudo kill -9 <PID>

# Restart service
sudo systemctl start rag-doc-processor
```

---

## 📊 Performance Metrics

### Expected Performance
- **Processing Rate**: 5-10 documents/minute
- **Memory Usage**: 1-2GB typical
- **CPU Usage**: 20-40% per document
- **Disk I/O**: Moderate during processing

### Monitor Performance
```bash
# CPU and memory
top -p $(systemctl show -p MainPID rag-doc-processor | cut -d= -f2)

# Disk I/O
iostat -x 1

# Processing rate (files per minute)
watch -n 60 'find /data/rag/incoming -type f | wc -l'
```

---

## 🔗 Integration Examples

### From CI/CD Pipeline
```yaml
# .gitlab-ci.yml or .github/workflows
deploy:
  script:
    - scp docs/*.pdf user@rag-server:/data/rag/incoming/drivers/
```

### From Network Share
```bash
# Cron job: Every hour
0 * * * * rsync -av /mnt/docs/ /data/rag/incoming/general/
```

### Via API
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@doc.pdf" \
  -F "domain=drivers"
```

---

## 📞 Need Help?

**Full Documentation**: `docs/Document-Processor-Daemon-Guide.md`

**Check Logs**:
```bash
# Service logs
sudo journalctl -u rag-doc-processor -n 100

# Application logs
sudo tail -f /var/log/rag/document_processor.log
```

**Common Log Locations**:
- Service logs: `journalctl -u rag-doc-processor`
- Application logs: `/var/log/rag/document_processor.log`
- System logs: `/var/log/syslog` or `/var/log/messages`

**Get Service Info**:
```bash
systemctl show rag-doc-processor
systemctl cat rag-doc-processor
```

---

**Quick Reference v1.0** | **Service**: rag-doc-processor | **Last Updated**: 2025-01-13
