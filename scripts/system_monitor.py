#!/usr/bin/env python3
"""
RAG System Monitor
Monitors system health, performance, and generates alerts
"""

import asyncio
import logging
import time
import json
import psutil
import subprocess
import requests
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add RAG system to path
sys.path.append('/opt/rag-system')

from core.rag_pipeline import RAGPipeline

class RAGSystemMonitor:
    def __init__(self):
        self.logger = self._setup_logging()
        self.rag_pipeline = None
        self.monitoring_active = True
        
        # Configuration
        self.config = {
            'check_interval': 60,  # seconds
            'api_url': 'http://localhost:8000',
            'alert_thresholds': {
                'cpu_percent': 90,
                'memory_percent': 85,
                'disk_percent': 90,
                'gpu_temp': 80,
                'response_time': 10,
                'error_rate': 0.1
            },
            'alert_email': {
                'enabled': False,  # Set to True to enable email alerts
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'from_email': 'rag-system@yourcompany.com',
                'to_email': 'admin@yourcompany.com'
            }
        }
        
        # Metrics history
        self.metrics_history = []
        self.alerts_history = []
        
        self.logger.info("RAG System Monitor initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/rag/system_monitor.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        self.logger.info("Starting RAG System Monitor...")
        
        # Initialize RAG pipeline connection
        try:
            self.rag_pipeline = RAGPipeline()
            self.logger.info("Connected to RAG pipeline")
        except Exception as e:
            self.logger.warning(f"Could not connect to RAG pipeline: {e}")
        
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Check for alerts
                alerts = self._check_alerts(metrics)
                
                # Process alerts
                if alerts:
                    await self._process_alerts(alerts, metrics)
                
                # Log status
                self._log_status(metrics)
                
                # Wait before next check
                await asyncio.sleep(self.config['check_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': self._get_system_metrics(),
            'gpu': self._get_gpu_metrics(),
            'services': await self._check_services(),
            'api': await self._check_api_health(),
            'database': await self._check_database(),
            'storage': self._get_storage_metrics(),
            'network': self._get_network_metrics()
        }
        
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Process metrics
            process_count = len(psutil.pids())
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'load_average': {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                },
                'memory': {
                    'total': memory.total,
                    'used': memory.used,
                    'free': memory.free,
                    'percent': memory.percent,
                    'available': memory.available
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                },
                'process_count': process_count
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def _get_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Get GPU metrics using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 9:
                            gpus.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_used': int(parts[2]),
                                'memory_total': int(parts[3]),
                                'utilization_gpu': int(parts[4]),
                                'utilization_memory': int(parts[5]),
                                'temperature': int(parts[6]),
                                'power_draw': float(parts[7]) if parts[7] != 'N/A' else 0,
                                'power_limit': float(parts[8]) if parts[8] != 'N/A' else 0
                            })
                return gpus
            else:
                return []
                
        except Exception as e:
            self.logger.warning(f"Could not get GPU metrics: {e}")
            return []
    
    async def _check_services(self) -> Dict[str, bool]:
        """Check status of critical services"""
        services = {
            'postgresql': False,
            'redis-rag': False,
            'qdrant': False,
            'rag-api': False,
            'rag-processor': False,
            'nginx': False
        }
        
        for service in services.keys():
            try:
                result = subprocess.run([
                    'systemctl', 'is-active', service
                ], capture_output=True, text=True)
                
                services[service] = result.stdout.strip() == 'active'
                
            except Exception as e:
                self.logger.warning(f"Could not check service {service}: {e}")
                services[service] = False
        
        return services
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API health and performance"""
        api_metrics = {
            'available': False,
            'response_time': None,
            'status_code': None,
            'error': None
        }
        
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.config['api_url']}/api/health",
                timeout=30
            )
            response_time = time.time() - start_time
            
            api_metrics.update({
                'available': True,
                'response_time': response_time,
                'status_code': response.status_code,
                'health_data': response.json() if response.status_code == 200 else None
            })
            
        except requests.RequestException as e:
            api_metrics.update({
                'available': False,
                'error': str(e)
            })
        
        return api_metrics
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database health"""
        db_metrics = {
            'postgresql': {'available': False},
            'qdrant': {'available': False},
            'redis': {'available': False}
        }
        
        # Check PostgreSQL
        try:
            result = subprocess.run([
                'sudo', '-u', 'rag-system', 'psql', 'rag_metadata', 
                '-c', 'SELECT 1;'
            ], capture_output=True, text=True, timeout=10)
            
            db_metrics['postgresql']['available'] = result.returncode == 0
            
            if result.returncode == 0:
                # Get database size
                size_result = subprocess.run([
                    'sudo', '-u', 'rag-system', 'psql', 'rag_metadata',
                    '-t', '-c', 'SELECT pg_database_size(current_database());'
                ], capture_output=True, text=True)
                
                if size_result.returncode == 0:
                    db_metrics['postgresql']['size_bytes'] = int(size_result.stdout.strip())
                    
        except Exception as e:
            self.logger.warning(f"Could not check PostgreSQL: {e}")
        
        # Check Qdrant
        try:
            response = requests.get(
                'http://localhost:6333/collections',
                timeout=10
            )
            db_metrics['qdrant']['available'] = response.status_code == 200
            
            if response.status_code == 200:
                collections = response.json()
                db_metrics['qdrant']['collections'] = collections
                
        except Exception as e:
            self.logger.warning(f"Could not check Qdrant: {e}")
        
        # Check Redis
        try:
            result = subprocess.run([
                'redis-cli', '-p', '6380', 'ping'
            ], capture_output=True, text=True, timeout=10)
            
            db_metrics['redis']['available'] = result.stdout.strip() == 'PONG'
            
            if db_metrics['redis']['available']:
                # Get memory usage
                info_result = subprocess.run([
                    'redis-cli', '-p', '6380', 'info', 'memory'
                ], capture_output=True, text=True)
                
                if info_result.returncode == 0:
                    memory_info = {}
                    for line in info_result.stdout.split('\n'):
                        if ':' in line and not line.startswith('#'):
                            key, value = line.strip().split(':', 1)
                            memory_info[key] = value
                    db_metrics['redis']['memory_info'] = memory_info
                    
        except Exception as e:
            self.logger.warning(f"Could not check Redis: {e}")
        
        return db_metrics
    
    def _get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage usage metrics"""
        storage_metrics = {}
        
        # Important directories to monitor
        directories = {
            'root': '/',
            'rag_data': '/data/rag',
            'rag_documents': '/data/rag/documents',
            'rag_vectors': '/data/rag/vectors',
            'rag_backups': '/data/rag/backups',
            'logs': '/var/log/rag'
        }
        
        for name, path in directories.items():
            try:
                if Path(path).exists():
                    usage = psutil.disk_usage(path)
                    storage_metrics[name] = {
                        'path': path,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': (usage.used / usage.total) * 100
                    }
            except Exception as e:
                self.logger.warning(f"Could not get storage metrics for {path}: {e}")
        
        return storage_metrics
    
    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get network metrics"""
        try:
            net_io = psutil.net_io_counters()
            net_connections = len(psutil.net_connections())
            
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'connections': net_connections
            }
        except Exception as e:
            self.logger.warning(f"Could not get network metrics: {e}")
            return {}
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics for history tracking"""
        self.metrics_history.append(metrics)
        
        # Keep only last 24 hours of metrics (assuming 1-minute intervals)
        if len(self.metrics_history) > 1440:
            self.metrics_history = self.metrics_history[-1440:]
        
        # Save to file
        try:
            metrics_file = Path('/var/log/rag/metrics_history.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history[-100:], f)  # Save last 100 entries
        except Exception as e:
            self.logger.warning(f"Could not save metrics history: {e}")
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        thresholds = self.config['alert_thresholds']
        
        # System alerts
        system = metrics.get('system', {})
        if system.get('cpu_percent', 0) > thresholds['cpu_percent']:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f"High CPU usage: {system['cpu_percent']:.1f}%",
                'value': system['cpu_percent']
            })
        
        if system.get('memory', {}).get('percent', 0) > thresholds['memory_percent']:
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning', 
                'message': f"High memory usage: {system['memory']['percent']:.1f}%",
                'value': system['memory']['percent']
            })
        
        # GPU alerts
        for gpu in metrics.get('gpu', []):
            if gpu.get('temperature', 0) > thresholds['gpu_temp']:
                alerts.append({
                    'type': 'high_gpu_temp',
                    'severity': 'warning',
                    'message': f"High GPU {gpu['index']} temperature: {gpu['temperature']}°C",
                    '