#!/usr/bin/env python3
"""
Configuration Validation Script for RAG System
Validates system configuration and dependencies
"""

import os
import sys
import json
import yaml
import psutil
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Any, Tuple
import importlib.util

class ConfigValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.config = {}
        
    def validate_all(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all validation checks"""
        print("🔍 Validating RAG System Configuration...")
        print("=" * 50)
        
        # Load configuration
        self._load_configuration()
        
        # Run validation checks
        self._validate_system_requirements()
        self._validate_directories()
        self._validate_dependencies()
        self._validate_services()
        self._validate_models()
        self._validate_database_connection()
        self._validate_gpu_setup()
        self._validate_permissions()
        
        # Generate report
        return self._generate_report()
    
    def _load_configuration(self):
        """Load and validate configuration file"""
        config_path = Path('/opt/rag-system/config/rag_config.yaml')
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                self.info.append(f"✓ Configuration loaded from {config_path}")
            else:
                self.errors.append(f"✗ Configuration file not found: {config_path}")
                # Load default config for validation
                self.config = self._get_default_config()
                
        except yaml.YAMLError as e:
            self.errors.append(f"✗ Invalid YAML in configuration file: {e}")
        except Exception as e:
            self.errors.append(f"✗ Error loading configuration: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "rag_metadata",
                "user": "rag-system"
            },
            "qdrant": {
                "host": "localhost", 
                "port": 6333
            },
            "redis": {
                "host": "localhost",
                "port": 6380,
                "db": 0
            },
            "language_model": {
                "provider": "ollama",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "deepseek-r1:70b"
                }
            }
        }
    
    def _validate_system_requirements(self):
        """Validate system requirements"""
        print("🖥️  Checking System Requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 11):
            self.info.append(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            self.errors.append(f"✗ Python version {python_version.major}.{python_version.minor} < 3.11 required")
        
        # Check system resources
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb >= 64:
            self.info.append(f"✓ System RAM: {memory_gb:.1f} GB")
        elif memory_gb >= 32:
            self.warnings.append(f"⚠ System RAM: {memory_gb:.1f} GB (64+ GB recommended)")
        else:
            self.errors.append(f"✗ System RAM: {memory_gb:.1f} GB (minimum 32 GB required)")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        disk_gb = disk.free / (1024**3)
        
        if disk_gb >= 100:
            self.info.append(f"✓ Free disk space: {disk_gb:.1f} GB")
        elif disk_gb >= 50:
            self.warnings.append(f"⚠ Free disk space: {disk_gb:.1f} GB (100+ GB recommended)")
        else:
            self.errors.append(f"✗ Free disk space: {disk_gb:.1f} GB (minimum 50 GB required)")
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count >= 16:
            self.info.append(f"✓ CPU cores: {cpu_count}")
        elif cpu_count >= 8:
            self.warnings.append(f"⚠ CPU cores: {cpu_count} (16+ recommended)")
    def _validate_directories(self):
        """Validate required directories"""
        print("📁 Checking Directory Structure...")
        
        required_dirs = [
            '/opt/rag-system',
            '/opt/rag-system/config',
            '/opt/rag-system/models',
            '/opt/rag-system/scripts',
            '/data/rag',
            '/data/rag/documents',
            '/data/rag/vectors',
            '/data/rag/backups',
            '/var/log/rag'
        ]
        
        for directory in required_dirs:
            path = Path(directory)
            if path.exists() and path.is_dir():
                # Check permissions
                if os.access(path, os.R_OK | os.W_OK):
                    self.info.append(f"✓ Directory: {directory}")
                else:
                    self.warnings.append(f"⚠ Directory exists but lacks permissions: {directory}")
            else:
                self.errors.append(f"✗ Missing directory: {directory}")
    
    def _validate_dependencies(self):
        """Validate Python dependencies"""
        print("📦 Checking Python Dependencies...")
        
        required_packages = [
            'fastapi',
            'uvicorn',
            'psycopg2',
            'redis',
            'qdrant-client',
            'sentence-transformers',
            'transformers',
            'torch',
            'pandas',
            'numpy',
            'nltk',
            'spacy',
            'langchain',
            'PyMuPDF',
            'python-docx',
            'unstructured',
            'tree-sitter',
            'watchdog',
            'schedule',
            'cryptography',
            'pyjwt'
        ]
        
        missing_packages = []
        outdated_packages = []
        
        for package in required_packages:
            try:
                spec = importlib.util.find_spec(package.replace('-', '_'))
                if spec is not None:
                    self.info.append(f"✓ Package: {package}")
                else:
                    missing_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.errors.append(f"✗ Missing packages: {', '.join(missing_packages)}")
        
        # Check for critical packages with version requirements
        version_checks = {
            'torch': '2.0.0',
            'transformers': '4.30.0',
            'fastapi': '0.100.0'
        }
        
        for package, min_version in version_checks.items():
            try:
                module = __import__(package)
                if hasattr(module, '__version__'):
                    current_version = module.__version__
                    self.info.append(f"✓ {package} version: {current_version}")
                else:
                    self.warnings.append(f"⚠ Could not determine {package} version")
            except ImportError:
                pass  # Already handled above
    
    def _validate_services(self):
        """Validate system services"""
        print("🔧 Checking System Services...")
        
        services = [
            ('postgresql', 'PostgreSQL Database'),
            ('redis-rag', 'Redis Cache'),
            ('qdrant', 'Qdrant Vector Database'),
            ('nginx', 'Nginx Web Server')
        ]
        
        for service, description in services:
            try:
                result = subprocess.run([
                    'systemctl', 'is-active', service
                ], capture_output=True, text=True)
                
                if result.stdout.strip() == 'active':
                    self.info.append(f"✓ Service running: {description}")
                else:
                    self.warnings.append(f"⚠ Service not active: {description}")
                    
                # Check if service is enabled
                result = subprocess.run([
                    'systemctl', 'is-enabled', service
                ], capture_output=True, text=True)
                
                if result.stdout.strip() == 'enabled':
                    self.info.append(f"✓ Service enabled: {description}")
                else:
                    self.warnings.append(f"⚠ Service not enabled: {description}")
                    
            except Exception as e:
                self.errors.append(f"✗ Error checking service {service}: {e}")
    
    def _validate_models(self):
        """Validate AI models"""
        print("🤖 Checking AI Models...")
        
        # Check embedding models
        models_dir = Path('/opt/rag-system/models')
        required_models = ['bge-m3', 'codebert-base', 'e5-large-v2']
        
        for model in required_models:
            model_path = models_dir / model
            if model_path.exists():
                self.info.append(f"✓ Embedding model: {model}")
            else:
                self.warnings.append(f"⚠ Missing embedding model: {model}")
        
        # Check Ollama models
        try:
            result = subprocess.run([
                'ollama', 'list'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                models = result.stdout
                llm_config = self.config.get('language_model', {}).get('ollama', {})
                expected_model = llm_config.get('model', 'deepseek-r1:70b')
                
                if expected_model in models:
                    self.info.append(f"✓ LLM model available: {expected_model}")
                else:
                    self.warnings.append(f"⚠ LLM model not found: {expected_model}")
                    self.info.append("Available models:")
                    for line in models.split('\n')[1:]:  # Skip header
                        if line.strip():
                            self.info.append(f"  - {line.split()[0]}")
            else:
                self.errors.append("✗ Could not list Ollama models")
                
        except FileNotFoundError:
            self.errors.append("✗ Ollama not found in PATH")
        except Exception as e:
            self.errors.append(f"✗ Error checking Ollama models: {e}")
    
    def _validate_database_connection(self):
        """Validate database connections"""
        print("🗄️  Checking Database Connections...")
        
        # PostgreSQL
        try:
            import psycopg2
            db_config = self.config.get('database', {})
            
            conn = psycopg2.connect(
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 5432),
                database=db_config.get('database', 'rag_metadata'),
                user=db_config.get('user', 'rag-system')
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            self.info.append(f"✓ PostgreSQL connection: {version.split()[0]} {version.split()[1]}")
            
            # Check required tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['documents', 'chunks', 'user_access', 'query_log']
            for table in required_tables:
                if table in tables:
                    self.info.append(f"✓ Database table: {table}")
                else:
                    self.warnings.append(f"⚠ Missing database table: {table}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.errors.append(f"✗ PostgreSQL connection failed: {e}")
        
        # Redis
        try:
            import redis
            redis_config = self.config.get('redis', {})
            
            r = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6380),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
            
            r.ping()
            info = r.info()
            self.info.append(f"✓ Redis connection: version {info['redis_version']}")
            
        except Exception as e:
            self.errors.append(f"✗ Redis connection failed: {e}")
        
        # Qdrant
        try:
            qdrant_config = self.config.get('qdrant', {})
            response = requests.get(
                f"http://{qdrant_config.get('host', 'localhost')}:{qdrant_config.get('port', 6333)}/collections",
                timeout=10
            )
            
            if response.status_code == 200:
                collections = response.json()
                self.info.append(f"✓ Qdrant connection: {len(collections.get('result', {}).get('collections', []))} collections")
            else:
                self.warnings.append(f"⚠ Qdrant responded with status {response.status_code}")
                
        except Exception as e:
            self.errors.append(f"✗ Qdrant connection failed: {e}")
    
    def _validate_gpu_setup(self):
        """Validate GPU setup"""
        print("🎮 Checking GPU Setup...")
        
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                gpus = []
                total_vram = 0
                
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            index, name, memory = parts[0], parts[1], int(parts[2])
                            gpus.append({'index': index, 'name': name, 'memory': memory})
                            total_vram += memory
                
                self.info.append(f"✓ GPUs detected: {len(gpus)}")
                for gpu in gpus:
                    self.info.append(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory']} MB)")
                
                self.info.append(f"✓ Total VRAM: {total_vram} MB ({total_vram/1024:.1f} GB)")
                
                # Check if enough VRAM for models
                if total_vram >= 40000:  # 40GB for large models
                    self.info.append("✓ Sufficient VRAM for large models (70B)")
                elif total_vram >= 24000:  # 24GB for medium models
                    self.warnings.append("⚠ VRAM suitable for medium models (13B-34B)")
                else:
                    self.warnings.append("⚠ Limited VRAM, consider smaller models (7B)")
                
                # Check CUDA
                try:
                    import torch
                    if torch.cuda.is_available():
                        cuda_version = torch.version.cuda
                        self.info.append(f"✓ CUDA available: version {cuda_version}")
                        self.info.append(f"✓ PyTorch CUDA devices: {torch.cuda.device_count()}")
                    else:
                        self.warnings.append("⚠ CUDA not available in PyTorch")
                except ImportError:
                    self.warnings.append("⚠ PyTorch not available for CUDA check")
                    
            else:
                self.warnings.append("⚠ nvidia-smi not available or no GPUs detected")
                
        except FileNotFoundError:
            self.warnings.append("⚠ nvidia-smi not found - GPU support may not be available")
        except Exception as e:
            self.errors.append(f"✗ Error checking GPU setup: {e}")
    
    def _validate_permissions(self):
        """Validate file permissions"""
        print("🔐 Checking Permissions...")
        
        # Check if running as correct user
        current_user = os.getenv('USER', 'unknown')
        if current_user == 'root':
            self.warnings.append("⚠ Running as root - consider using rag-system user")
        elif current_user == 'rag-system':
            self.info.append("✓ Running as rag-system user")
        else:
            self.warnings.append(f"⚠ Running as {current_user} - may have permission issues")
        
        # Check critical file permissions
        critical_files = [
            '/opt/rag-system/config/rag_config.yaml',
            '/data/rag',
            '/var/log/rag'
        ]
        
        for file_path in critical_files:
            path = Path(file_path)
            if path.exists():
                if os.access(path, os.R_OK | os.W_OK):
                    self.info.append(f"✓ Permissions OK: {file_path}")
                else:
                    self.errors.append(f"✗ Permission denied: {file_path}")
            # Already checked existence in directory validation
    
    def _generate_report(self) -> Tuple[bool, Dict[str, Any]]:
        """Generate validation report"""
        print("\n" + "=" * 50)
        print("📊 VALIDATION REPORT")
        print("=" * 50)
        
        # Summary
        total_checks = len(self.info) + len(self.warnings) + len(self.errors)
        success_rate = len(self.info) / total_checks * 100 if total_checks > 0 else 0
        
        print(f"Total Checks: {total_checks}")
        print(f"✓ Passed: {len(self.info)}")
        print(f"⚠ Warnings: {len(self.warnings)}")
        print(f"✗ Errors: {len(self.errors)}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        # Show details
        if self.errors:
            print("🚨 ERRORS (Must Fix):")
            for error in self.errors:
                print(f"  {error}")
            print()
        
        if self.warnings:
            print("⚠️  WARNINGS (Should Fix):")
            for warning in self.warnings:
                print(f"  {warning}")
            print()
        
        if self.info:
            print("✅ PASSED CHECKS:")
            for info in self.info:
                print(f"  {info}")
            print()
        
        # Overall status
        is_valid = len(self.errors) == 0
        
        if is_valid:
            if len(self.warnings) == 0:
                print("🎉 SYSTEM READY: All checks passed!")
            else:
                print("✅ SYSTEM FUNCTIONAL: No critical errors, but some warnings exist.")
        else:
            print("❌ SYSTEM NOT READY: Critical errors must be fixed before deployment.")
        
        # Recommendations
        print("\n📋 RECOMMENDATIONS:")
        
        if self.errors:
            print("1. Fix all critical errors listed above")
            print("2. Run validation again after fixes")
        
        if self.warnings:
            print("3. Address warnings for optimal performance")
        
        if is_valid:
            print("4. System is ready for deployment!")
            print("5. Run 'sudo /opt/rag-system/scripts/rag-control.sh start' to start services")
        
        # Generate JSON report
        report = {
            'timestamp': str(datetime.now()),
            'status': 'ready' if is_valid else 'not_ready',
            'summary': {
                'total_checks': total_checks,
                'passed': len(self.info),
                'warnings': len(self.warnings),
                'errors': len(self.errors),
                'success_rate': success_rate
            },
            'details': {
                'passed': self.info,
                'warnings': self.warnings,
                'errors': self.errors
            },
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': os.uname().sysname if hasattr(os, 'uname') else 'Unknown',
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
            }
        }
        
        # Save report to file
        report_file = Path('/var/log/rag/validation_report.json')
        try:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save report: {e}")
        
        return is_valid, report

def main():
    """Main validation entry point"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Validate RAG System Configuration')
    parser.add_argument('--json', action='store_true', help='Output JSON report only')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    
    if args.quiet:
        # Redirect stdout to suppress print statements during validation
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            is_valid, report = validator.validate_all()
    else:
        is_valid, report = validator.validate_all()
    
    if args.json:
        print(json.dumps(report, indent=2))
    
    return 0 if is_valid else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())