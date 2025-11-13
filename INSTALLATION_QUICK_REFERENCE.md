# RAG System - Installation Quick Reference Card

## 🚀 Quick Commands

### Fresh Installation
```bash
sudo ./install_rag_system.sh --install
```

### Resume After Failure
```bash
sudo ./install_rag_system.sh --resume
```

### Check Status
```bash
sudo ./install_rag_system.sh --status
```

### Start from Specific Stage
```bash
sudo ./install_rag_system.sh --stage 9  # e.g., start from stage 9
```

### List All Stages
```bash
sudo ./install_rag_system.sh --list-stages
```

### Validate Installation
```bash
sudo ./install_rag_system.sh --validate
```

### Show Help
```bash
sudo ./install_rag_system.sh --help
```

---

## 🔄 Rollback Commands

### List Backups
```bash
sudo ./rollback_installation.sh --list
```

### Auto Rollback (to last backup)
```bash
sudo ./rollback_installation.sh --auto
```

### Rollback to Specific Backup
```bash
sudo ./rollback_installation.sh --file /opt/rag-system-backup/backup-file.tar.gz
```

---

## 📊 17 Installation Stages

| # | Stage | Time | Description |
|---|-------|------|-------------|
| 01 | preflight_checks | 1-2m | Verify system requirements |
| 02 | backup_existing | 2-5m | Backup current installation |
| 03 | install_system_deps | 5-10m | Install OS packages |
| 04 | setup_python_env | 2-3m | Create virtual environment |
| 05 | install_python_deps | 10-15m | Install Python packages |
| 06 | setup_database | 2-3m | Configure PostgreSQL |
| 07 | setup_redis | 1-2m | Configure Redis |
| 08 | setup_qdrant | 2-3m | Install vector database |
| 09 | download_models | 30-60m | Download ML models ⏱️ |
| 10 | setup_directories | 1m | Create directories |
| 11 | copy_application_files | 1-2m | Copy code files |
| 12 | setup_configuration | 1-2m | Configure system |
| 13 | setup_systemd_services | 1m | Create services |
| 14 | initialize_database | 2-3m | Create DB schema |
| 15 | run_migrations | 1m | Apply migrations |
| 16 | validate_installation | 2-3m | Run validation |
| 17 | start_services | 2-3m | Start all services |

**Total Time**: 60-90 minutes

---

## ⚡ Common Failure Recovery

### Stage 5 Failed (Python Dependencies)
```bash
# Clear pip cache
rm -rf ~/.cache/pip

# Resume
sudo ./install_rag_system.sh --resume
```

### Stage 9 Failed (Model Download)
```bash
# Check internet
ping -c 3 huggingface.co

# If behind proxy
export HTTP_PROXY=http://proxy:8080
export HTTPS_PROXY=http://proxy:8080

# Resume
sudo -E ./install_rag_system.sh --resume
```

### Stage 14 Failed (Database)
```bash
# Restart PostgreSQL
sudo systemctl restart postgresql

# Resume
sudo ./install_rag_system.sh --resume
```

### Service Won't Start
```bash
# Check logs
sudo journalctl -u rag-api -n 50

# Restart service
sudo systemctl restart rag-api
```

---

## 🔍 Post-Installation Checks

### Verify Services
```bash
# Check all services
sudo systemctl status rag-api
sudo systemctl status rag-doc-processor
sudo systemctl status postgresql
sudo systemctl status redis-server
sudo systemctl status qdrant
```

### Test API
```bash
# Health check
curl http://localhost:8000/api/health

# API docs
open http://localhost:8000/api/docs
```

### View Logs
```bash
# Installation logs
tail -f /var/log/rag-install/install_*.log

# Service logs
sudo journalctl -u rag-api -f
```

---

## 📝 Important File Locations

| Item | Location |
|------|----------|
| Installation | `/opt/rag-system/` |
| Configuration | `/opt/rag-system/config/rag_config.yaml` |
| Environment | `/opt/rag-system/.env` |
| Models | `/opt/rag-system/models/` |
| Documents | `/data/rag/documents/` |
| Logs | `/var/log/rag/` |
| Install Logs | `/var/log/rag-install/` |
| State File | `/var/lib/rag-system/install_state.json` |
| Backups | `/opt/rag-system-backup/` |

---

## 🛠️ Troubleshooting Quick Fixes

### "Permission denied"
```bash
# Check you're running as root
sudo su -
```

### "Python version too old"
```bash
# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

### "Insufficient disk space"
```bash
# Check space
df -h /

# Clean up
sudo apt clean
sudo apt autoremove
```

### "Database connection failed"
```bash
# Check PostgreSQL
sudo systemctl status postgresql

# Restart if needed
sudo systemctl restart postgresql
```

### "Cannot connect to Redis"
```bash
# Check Redis
sudo systemctl status redis-server

# Test connection
redis-cli -p 6380 ping
```

---

## 🎯 Production Checklist

### Before Installation
- [ ] System meets minimum requirements (32GB RAM, 100GB disk)
- [ ] Python 3.11+ installed
- [ ] Internet connectivity available (for downloads)
- [ ] Root/sudo access
- [ ] Backup current system if upgrading

### After Installation
- [ ] All services running (`systemctl status`)
- [ ] API health check passes (`curl localhost:8000/api/health`)
- [ ] Can access API docs (`http://localhost:8000/api/docs`)
- [ ] Admin user created
- [ ] Sample document ingestion tested
- [ ] Firewall rules configured (if needed)
- [ ] SSL/TLS certificates installed (production)
- [ ] Monitoring configured

### Security
- [ ] Change default passwords in `/opt/rag-system/.env`
- [ ] Configure firewall (ufw/iptables)
- [ ] Setup SSL certificates
- [ ] Enable audit logging
- [ ] Configure user clearance levels
- [ ] Setup backup schedule

---

## 📞 Need Help?

### Documentation
- Full Guide: `docs/Installation-Guide.md`
- Improvements: `docs/RAG-System-Improvements-2025.md`
- Features: `README_NEW_FEATURES.md`
- Architecture: `docs/RAG-System-Architecture.md`

### Logs
```bash
# Installation logs
less /var/log/rag-install/install_$(ls -t /var/log/rag-install/ | head -1)

# Service logs
sudo journalctl -u rag-api --since "1 hour ago"
```

### Support Commands
```bash
# Get system info
uname -a
lsb_release -a
df -h
free -h
nvidia-smi  # If GPU

# Get installation status
sudo ./install_rag_system.sh --status

# Validate system
sudo ./install_rag_system.sh --validate
```

---

## 💡 Pro Tips

1. **Stage 9 (Model Download)** is the longest - be patient or run overnight
2. **Always check logs** if something fails: `/var/log/rag-install/`
3. **Use `--resume`** liberally - it's safe and picks up where it left off
4. **Create manual backups** before major changes
5. **Monitor disk space** during installation (especially stage 9)
6. **Test connectivity** to HuggingFace before starting if behind firewall
7. **Keep a copy** of the backup file path from stage 2

---

**Quick Reference Card v1.0**
**For Installation Script v2.0**
**Last Updated**: 2025-01-13
