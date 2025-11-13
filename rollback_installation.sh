#!/bin/bash

################################################################################
# RAG System - Rollback Script
# Rollsback to previous backup
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_DIR="/opt/rag-system"
BACKUP_DIR="/opt/rag-system-backup"
STATE_FILE="/var/lib/rag-system/install_state.json"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $@"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $@"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $@"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run as root or with sudo"
        exit 1
    fi
}

get_backup_path() {
    if [ -f "${STATE_FILE}" ]; then
        python3 -c "import json; print(json.load(open('${STATE_FILE}'))['backup_path'])" 2>/dev/null || echo ""
    else
        echo ""
    fi
}

list_backups() {
    echo "Available backups:"
    echo "=================="

    if [ -d "${BACKUP_DIR}" ]; then
        ls -lh "${BACKUP_DIR}"/*.tar.gz 2>/dev/null | awk '{print $9, "(" $5 ")"}'
    else
        echo "No backups found"
    fi
}

perform_rollback() {
    local backup_file=$1

    if [ ! -f "${backup_file}" ]; then
        log_error "Backup file not found: ${backup_file}"
        return 1
    fi

    log_warning "This will stop all RAG services and restore from backup"
    read -p "Are you sure you want to continue? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        log_info "Rollback cancelled"
        return 0
    fi

    # Stop services
    log_info "Stopping RAG services..."
    systemctl stop rag-api 2>/dev/null || true
    systemctl stop rag-doc-processor 2>/dev/null || true

    # Backup current state before rollback
    if [ -d "${INSTALL_DIR}" ]; then
        local pre_rollback_backup="${BACKUP_DIR}/pre-rollback-$(date +%Y%m%d_%H%M%S).tar.gz"
        log_info "Creating pre-rollback backup: ${pre_rollback_backup}"
        tar -czf "${pre_rollback_backup}" -C "$(dirname ${INSTALL_DIR})" "$(basename ${INSTALL_DIR})"
    fi

    # Remove current installation
    log_info "Removing current installation..."
    rm -rf "${INSTALL_DIR}"

    # Restore from backup
    log_info "Restoring from backup: ${backup_file}"
    tar -xzf "${backup_file}" -C "$(dirname ${INSTALL_DIR})"

    # Restart services
    log_info "Starting RAG services..."
    systemctl start rag-api
    systemctl start rag-doc-processor

    # Wait and check
    sleep 5

    if systemctl is-active --quiet rag-api; then
        log_info "✓ RAG API service is running"
    else
        log_warning "⚠ RAG API service failed to start"
    fi

    log_info "Rollback completed"
    return 0
}

main() {
    check_root

    case "${1:-}" in
        --list)
            list_backups
            ;;
        --auto)
            # Use backup from state file
            local backup_path=$(get_backup_path)
            if [ -z "$backup_path" ]; then
                log_error "No backup path found in state file"
                exit 1
            fi
            perform_rollback "$backup_path"
            ;;
        --file)
            if [ -z "$2" ]; then
                log_error "Please specify backup file"
                exit 1
            fi
            perform_rollback "$2"
            ;;
        *)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --list          List available backups"
            echo "  --auto          Rollback using backup from last installation"
            echo "  --file FILE     Rollback using specific backup file"
            echo ""
            echo "Examples:"
            echo "  $0 --list"
            echo "  $0 --auto"
            echo "  $0 --file /opt/rag-system-backup/rag-system-backup-20250113.tar.gz"
            exit 1
            ;;
    esac
}

main "$@"
