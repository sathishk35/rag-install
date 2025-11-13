#!/usr/bin/env python3
"""
Comprehensive Backup Script for RAG System
Backs up databases, configurations, models, and documents
"""

import os
import sys
import gzip
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import asyncio
import tarfile
import hashlib
from typing import Dict, List, Any, Optional

class RAGSystemBackup:
    def __init__(self, config_path: str = "/opt/rag-system/config/rag_config.yaml"):
        self.logger = self._setup_logging()
        self.backup_base_dir = Path('/data/rag/backups')
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Backup configuration
        self.backup_config = {
            'retention_days': 30,
            'compress': True,
            'verify_backups': True,
            'backup_types': {
                'full': ['database', 'vectors', 'documents', 'config', 'models'],
                'data': ['database', 'vectors', 'documents'],
                'config': ['config', 'models'],
                'database': ['database'],
                'minimal': ['database', 'config']
            }
        }
        
        # Ensure backup directory exists
        self.backup_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"RAG System Backup initialized - Config: {self.config_path}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("/var/log/rag")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'backup.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_config(self):
        """Load system configuration"""
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
                
        # Default configuration
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "rag_metadata",
                "user": "rag-system"
            },
            "data": {
                "storage": {
                    "base_directory": "/data/rag",
                    "documents_directory": "/data/rag/documents",
                    "vectors_directory": "/data/rag/vectors",
                    "models_directory": "/opt/rag-system/models"
                }
            }
        }
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    async def create_backup(self, backup_type: str = 'full', custom_name: str = None) -> Dict[str, Any]:
        """Create a system backup"""
        
        if backup_type not in self.backup_config['backup_types']:
            raise ValueError(f"Invalid backup type: {backup_type}")
        
        # Create backup directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = custom_name or f"rag_backup_{backup_type}_{timestamp}"
        backup_dir = self.backup_base_dir / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting {backup_type} backup: {backup_name}")
        
        # Initialize backup metadata
        backup_metadata = {
            'backup_name': backup_name,
            'backup_type': backup_type,
            'timestamp': timestamp,
            'start_time': datetime.now().isoformat(),
            'components': self.backup_config['backup_types'][backup_type],
            'status': 'in_progress',
            'results': {},
            'total_size_bytes': 0,
            'compressed': self.backup_config['compress'],
            'config_path': str(self.config_path),
            'system_info': self._get_system_info()
        }
        
        try:
            # Backup each component
            for component in backup_metadata['components']:
                self.logger.info(f"Backing up component: {component}")
                result = await self._backup_component(component, backup_dir)
                backup_metadata['results'][component] = result
                backup_metadata['total_size_bytes'] += result.get('size_bytes', 0)
            
            # Create backup manifest
            manifest_path = backup_dir / 'backup_manifest.json'
            backup_metadata['end_time'] = datetime.now().isoformat()
            backup_metadata['status'] = 'completed'
            
            with open(manifest_path, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            # Create checksums file
            await self._create_checksums_file(backup_dir)
            
            # Verify backup if enabled
            if self.backup_config['verify_backups']:
                verification_result = await self._verify_backup(backup_dir)
                backup_metadata['verification'] = verification_result
                
                # Update manifest with verification results
                with open(manifest_path, 'w') as f:
                    json.dump(backup_metadata, f, indent=2)
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
            self.logger.info(f"Backup completed successfully: {backup_name}")
            self.logger.info(f"Total backup size: {backup_metadata['total_size_bytes'] / (1024**3):.2f} GB")
            
            return backup_metadata
            
        except Exception as e:
            backup_metadata['status'] = 'failed'
            backup_metadata['error'] = str(e)
            backup_metadata['end_time'] = datetime.now().isoformat()
            
            # Save failed backup metadata
            try:
                with open(backup_dir / 'backup_manifest.json', 'w') as f:
                    json.dump(backup_metadata, f, indent=2)
            except:
                pass
            
            self.logger.error(f"Backup failed: {e}")
            raise
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for backup metadata"""
        try:
            import platform
            import psutil
            
            return {
                'hostname': platform.node(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_usage': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free
                }
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system info: {e}")
            return {}
    
    async def _backup_component(self, component: str, backup_dir: Path) -> Dict[str, Any]:
        """Backup a specific component"""
        
        component_dir = backup_dir / component
        component_dir.mkdir(exist_ok=True)
        
        result = {
            'component': component,
            'start_time': datetime.now().isoformat(),
            'status': 'success',
            'size_bytes': 0,
            'files_count': 0,
            'checksum': ''
        }
        
        try:
            if component == 'database':
                result.update(await self._backup_database(component_dir))
                
            elif component == 'vectors':
                result.update(await self._backup_vectors(component_dir))
                
            elif component == 'documents':
                result.update(await self._backup_documents(component_dir))
                
            elif component == 'config':
                result.update(await self._backup_config(component_dir))
                
            elif component == 'models':
                result.update(await self._backup_models(component_dir))
                
            else:
                raise ValueError(f"Unknown component: {component}")
            
            result['end_time'] = datetime.now().isoformat()
            self.logger.info(f"Component {component} backed up: {result['size_bytes'] / (1024**2):.2f} MB")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['end_time'] = datetime.now().isoformat()
            self.logger.error(f"Failed to backup component {component}: {e}")
            raise
        
        return result
    
    async def _backup_database(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup PostgreSQL database"""
        
        db_config = self.config.get('database', {})
        
        # PostgreSQL backup
        pg_backup_file = backup_dir / 'postgresql_backup.sql'
        pg_compressed_file = backup_dir / 'postgresql_backup.sql.gz'
        
        # Create database dump
        cmd = [
            'pg_dump',
            '-h', db_config.get('host', 'localhost'),
            '-p', str(db_config.get('port', 5432)),
            '-U', db_config.get('user', 'rag-system'),
            '-d', db_config.get('database', 'rag_metadata'),
            '--no-password',
            '--verbose',
            '--format=plain',
            '--file', str(pg_backup_file)
        ]
        
        # Set environment for password-less authentication
        env = os.environ.copy()
        env['PGPASSFILE'] = '/home/rag-system/.pgpass'
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"PostgreSQL backup failed: {stderr.decode()}")
        
        # Get additional database info
        db_info = await self._get_database_info(db_config)
        
        # Compress backup
        final_file = pg_backup_file
        if self.backup_config['compress']:
            with open(pg_backup_file, 'rb') as f_in:
                with gzip.open(pg_compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            pg_backup_file.unlink()  # Remove uncompressed file
            final_file = pg_compressed_file
        
        return {
            'database_type': 'postgresql',
            'backup_file': str(final_file.name),
            'size_bytes': final_file.stat().st_size,
            'files_count': 1,
            'compressed': self.backup_config['compress'],
            'checksum': self._calculate_checksum(final_file),
            'database_info': db_info
        }
    
    async def _get_database_info(self, db_config: Dict) -> Dict[str, Any]:
        """Get database information for backup metadata"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 5432),
                database=db_config.get('database', 'rag_metadata'),
                user=db_config.get('user', 'rag-system')
            )
            
            cursor = conn.cursor()
            
            # Get table counts
            cursor.execute("""
                SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del 
                FROM pg_stat_user_tables
            """)
            tables_info = cursor.fetchall()
            
            # Get database size
            cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
            db_size = cursor.fetchone()[0]
            
            # Get document and chunk counts
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM user_access")
            user_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'database_size': db_size,
                'tables_info': tables_info,
                'document_count': doc_count,
                'chunk_count': chunk_count,
                'user_count': user_count
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get database info: {e}")
            return {}
    
    async def _backup_vectors(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup Qdrant vector database"""
        
        vectors_source = Path(self.config.get('data', {}).get('storage', {}).get('vectors_directory', '/data/rag/vectors'))
        
        if not vectors_source.exists():
            raise Exception(f"Vector database directory not found: {vectors_source}")
        
        # Create vector backup using rsync for efficiency
        cmd = [
            'rsync',
            '-av',
            '--progress',
            str(vectors_source) + '/',
            str(backup_dir) + '/'
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Vector backup failed: {stderr.decode()}")
        
        # Calculate total size and file count
        total_size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
        files_count = len(list(backup_dir.rglob('*')))
        
        # Get Qdrant collection info
        vector_info = await self._get_vector_info()
        
        # Compress if enabled
        compressed_file = None
        if self.backup_config['compress']:
            compressed_file = backup_dir.parent / f"{backup_dir.name}_vectors.tar.gz"
            
            with tarfile.open(compressed_file, 'w:gz') as tar:
                tar.add(backup_dir, arcname=backup_dir.name)
            
            # Remove uncompressed directory
            shutil.rmtree(backup_dir)
            total_size = compressed_file.stat().st_size
            files_count = 1
        
        final_path = compressed_file if compressed_file else backup_dir
        
        return {
            'source_directory': str(vectors_source),
            'backup_path': str(final_path),
            'size_bytes': total_size,
            'files_count': files_count,
            'compressed': self.backup_config['compress'],
            'checksum': self._calculate_checksum(compressed_file) if compressed_file else '',
            'vector_info': vector_info
        }
    
    async def _get_vector_info(self) -> Dict[str, Any]:
        """Get vector database information"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:6333/collections", timeout=10)
                if response.status_code == 200:
                    collections_data = response.json()
                    
                    collections_info = []
                    for collection in collections_data.get('result', {}).get('collections', []):
                        coll_name = collection['name']
                        
                        # Get detailed collection info
                        coll_response = await client.get(f"http://localhost:6333/collections/{coll_name}")
                        if coll_response.status_code == 200:
                            coll_data = coll_response.json()
                            collections_info.append({
                                'name': coll_name,
                                'vectors_count': coll_data.get('result', {}).get('vectors_count', 0),
                                'indexed_vectors_count': coll_data.get('result', {}).get('indexed_vectors_count', 0),
                                'points_count': coll_data.get('result', {}).get('points_count', 0)
                            })
                    
                    return {
                        'collections_count': len(collections_info),
                        'collections': collections_info,
                        'total_vectors': sum(c['vectors_count'] for c in collections_info)
                    }
                    
        except Exception as e:
            self.logger.warning(f"Failed to get vector info: {e}")
            
        return {}
    
    async def _backup_documents(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup document files"""
        
        documents_source = Path(self.config.get('data', {}).get('storage', {}).get('documents_directory', '/data/rag/documents'))
        
        if not documents_source.exists():
            raise Exception(f"Documents directory not found: {documents_source}")
        
        # Create documents backup using rsync
        cmd = [
            'rsync',
            '-av',
            '--progress',
            str(documents_source) + '/',
            str(backup_dir) + '/'
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Documents backup failed: {stderr.decode()}")
        
        # Calculate statistics
        total_size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
        files_count = len(list(backup_dir.rglob('*')))
        
        # Get document statistics by type
        doc_stats = self._analyze_documents(backup_dir)
        
        # Compress if enabled
        compressed_file = None
        if self.backup_config['compress']:
            compressed_file = backup_dir.parent / f"{backup_dir.name}_documents.tar.gz"
            
            with tarfile.open(compressed_file, 'w:gz') as tar:
                tar.add(backup_dir, arcname=backup_dir.name)
            
            # Remove uncompressed directory
            shutil.rmtree(backup_dir)
            total_size = compressed_file.stat().st_size
            files_count = 1
        
        final_path = compressed_file if compressed_file else backup_dir
        
        return {
            'source_directory': str(documents_source),
            'backup_path': str(final_path),
            'size_bytes': total_size,
            'files_count': files_count,
            'compressed': self.backup_config['compress'],
            'checksum': self._calculate_checksum(compressed_file) if compressed_file else '',
            'document_statistics': doc_stats
        }
    
    def _analyze_documents(self, docs_dir: Path) -> Dict[str, Any]:
        """Analyze document statistics"""
        try:
            stats = {
                'by_extension': {},
                'by_domain': {},
                'total_files': 0,
                'total_size': 0
            }
            
            for file_path in docs_dir.rglob('*'):
                if file_path.is_file():
                    stats['total_files'] += 1
                    file_size = file_path.stat().st_size
                    stats['total_size'] += file_size
                    
                    # Count by extension
                    ext = file_path.suffix.lower()
                    stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1
                    
                    # Count by domain (based on path)
                    path_parts = file_path.parts
                    if len(path_parts) > 1:
                        domain = path_parts[1] if path_parts[1] != 'documents' else path_parts[2] if len(path_parts) > 2 else 'general'
                        stats['by_domain'][domain] = stats['by_domain'].get(domain, 0) + 1
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze documents: {e}")
            return {}
    
    async def _backup_config(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup configuration files"""
        
        config_sources = [
            '/opt/rag-system/config',
            '/etc/systemd/system/rag-*.service',
            '/etc/nginx/sites-available/rag*',
            '/etc/logrotate.d/rag*',
            '/home/rag-system/.pgpass'
        ]
        
        total_size = 0
        files_count = 0
        backed_up_files = []
        
        for source in config_sources:
            source_path = Path(source)
            
            if '*' in source:
                # Handle glob patterns
                import glob
                for file_path in glob.glob(source):
                    file_path = Path(file_path)
                    if file_path.exists():
                        dest_path = backup_dir / file_path.name
                        shutil.copy2(file_path, dest_path)
                        total_size += dest_path.stat().st_size
                        files_count += 1
                        backed_up_files.append(str(file_path))
                        
            elif source_path.exists():
                if source_path.is_file():
                    dest_path = backup_dir / source_path.name
                    shutil.copy2(source_path, dest_path)
                    total_size += dest_path.stat().st_size
                    files_count += 1
                    backed_up_files.append(str(source_path))
                    
                elif source_path.is_dir():
                    dest_path = backup_dir / source_path.name
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    total_size += sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
                    files_count += len(list(dest_path.rglob('*')))
                    backed_up_files.append(str(source_path))
        
        # Create system info file
        system_info_file = backup_dir / 'system_info.json'
        with open(system_info_file, 'w') as f:
            json.dump(self._get_system_info(), f, indent=2)
        total_size += system_info_file.stat().st_size
        files_count += 1
        
        # Compress if enabled
        compressed_file = None
        if self.backup_config['compress'] and files_count > 0:
            compressed_file = backup_dir.parent / f"{backup_dir.name}_config.tar.gz"
            
            with tarfile.open(compressed_file, 'w:gz') as tar:
                tar.add(backup_dir, arcname=backup_dir.name)
            
            # Remove uncompressed directory
            shutil.rmtree(backup_dir)
            total_size = compressed_file.stat().st_size
            files_count = 1
        
        final_path = compressed_file if compressed_file else backup_dir
        
        return {
            'source_files': backed_up_files,
            'backup_path': str(final_path),
            'size_bytes': total_size,
            'files_count': files_count,
            'compressed': self.backup_config['compress'],
            'checksum': self._calculate_checksum(compressed_file) if compressed_file else ''
        }
    
    async def _backup_models(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup AI models"""
        
        models_source = Path(self.config.get('data', {}).get('storage', {}).get('models_directory', '/opt/rag-system/models'))
        
        if not models_source.exists():
            self.logger.warning(f"Models directory not found: {models_source}, skipping model backup")
            return {
                'source_directory': str(models_source),
                'size_bytes': 0,
                'files_count': 0,
                'compressed': False,
                'skipped': True,
                'reason': 'Models directory not found'
            }
        
        # Get model information
        model_info = self._get_model_info(models_source)
        
        # Models can be very large, so we'll create a compressed archive directly
        compressed_file = backup_dir.parent / f"{backup_dir.name}_models.tar.gz"
        
        cmd = [
            'tar',
            '-czf',
            str(compressed_file),
            '-C',
            str(models_source.parent),
            models_source.name
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Models backup failed: {stderr.decode()}")
        
        # Remove the empty backup directory since we created a compressed file directly
        if backup_dir.exists():
            backup_dir.rmdir()
        
        return {
            'source_directory': str(models_source),
            'backup_file': compressed_file.name,
            'backup_path': str(compressed_file),
            'size_bytes': compressed_file.stat().st_size,
            'files_count': 1,
            'compressed': True,
            'checksum': self._calculate_checksum(compressed_file),
            'model_info': model_info
        }
    
    def _get_model_info(self, models_dir: Path) -> Dict[str, Any]:
        """Get information about available models"""
        try:
            model_info = {
                'embedding_models': [],
                'language_models': [],
                'total_size': 0,
                'model_count': 0
            }
            
            for model_path in models_dir.iterdir():
                if model_path.is_dir():
                    model_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                    model_info['total_size'] += model_size
                    model_info['model_count'] += 1
                    
                    # Categorize models
                    model_name = model_path.name.lower()
                    if any(embed_type in model_name for embed_type in ['e5', 'bge', 'codebert', 'multilingual']):
                        model_info['embedding_models'].append({
                            'name': model_path.name,
                            'size_bytes': model_size,
                            'size_mb': round(model_size / (1024**2), 2)
                        })
                    else:
                        model_info['language_models'].append({
                            'name': model_path.name,
                            'size_bytes': model_size,
                            'size_mb': round(model_size / (1024**2), 2)
                        })
            
            return model_info
            
        except Exception as e:
            self.logger.warning(f"Failed to get model info: {e}")
            return {}
    
    async def _create_checksums_file(self, backup_dir: Path):
        """Create checksums file for all backup files"""
        checksums_file = backup_dir / 'checksums.md5'
        
        try:
            with open(checksums_file, 'w') as f:
                for file_path in backup_dir.rglob('*'):
                    if file_path.is_file() and file_path.name != 'checksums.md5':
                        checksum = self._calculate_checksum(file_path)
                        relative_path = file_path.relative_to(backup_dir)
                        f.write(f"{checksum}  {relative_path}\n")
                        
        except Exception as e:
            self.logger.warning(f"Failed to create checksums file: {e}")
    
    async def _verify_backup(self, backup_dir: Path) -> Dict[str, Any]:
        """Verify backup integrity"""
        
        verification_result = {
            'verified': True,
            'checks': [],
            'errors': [],
            'file_checks': 0,
            'checksum_matches': 0
        }
        
        try:
            # Check manifest file exists
            manifest_path = backup_dir / 'backup_manifest.json'
            if manifest_path.exists():
                verification_result['checks'].append('Manifest file exists')
                
                # Verify manifest is valid JSON
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    verification_result['checks'].append('Manifest is valid JSON')
                except json.JSONDecodeError as e:
                    verification_result['errors'].append(f'Invalid manifest JSON: {e}')
                    verification_result['verified'] = False
            else:
                verification_result['errors'].append('Manifest file missing')
                verification_result['verified'] = False
            
            # Verify checksums if available
            checksums_file = backup_dir / 'checksums.md5'
            if checksums_file.exists():
                verification_result['checks'].append('Checksums file exists')
                
                with open(checksums_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            expected_checksum, file_path = line.strip().split('  ', 1)
                            actual_file = backup_dir / file_path
                            
                            if actual_file.exists():
                                verification_result['file_checks'] += 1
                                actual_checksum = self._calculate_checksum(actual_file)
                                
                                if actual_checksum == expected_checksum:
                                    verification_result['checksum_matches'] += 1
                                else:
                                    verification_result['errors'].append(f'Checksum mismatch: {file_path}')
                                    verification_result['verified'] = False
                            else:
                                verification_result['errors'].append(f'Missing file: {file_path}')
                                verification_result['verified'] = False
            
            # Verify component backups exist and have content
            for component_file in backup_dir.iterdir():
                if component_file.is_file() and component_file.name not in ['backup_manifest.json', 'checksums.md5']:
                    # Check file size
                    if component_file.stat().st_size > 0:
                        verification_result['checks'].append(f'Component {component_file.name} has content')
                    else:
                        verification_result['errors'].append(f'Component {component_file.name} is empty')
                        verification_result['verified'] = False
                    
                    # Test compressed files
                    if component_file.name.endswith('.gz'):
                        try:
                            with gzip.open(component_file, 'rb') as f:
                                f.read(1024)  # Try to read first 1KB
                            verification_result['checks'].append(f'Compressed file {component_file.name} is readable')
                        except Exception as e:
                            verification_result['errors'].append(f'Compressed file {component_file.name} is corrupted: {e}')
                            verification_result['verified'] = False
                    
                    # Test tar.gz files
                    elif component_file.name.endswith('.tar.gz'):
                        try:
                            with tarfile.open(component_file, 'r:gz') as tar:
                                tar.getnames()[:5]  # Try to read first 5 entries
                            verification_result['checks'].append(f'Archive {component_file.name} is readable')
                        except Exception as e:
                            verification_result['errors'].append(f'Archive {component_file.name} is corrupted: {e}')
                            verification_result['verified'] = False
            
            self.logger.info(f"Backup verification: {'PASSED' if verification_result['verified'] else 'FAILED'}")
            
        except Exception as e:
            verification_result['verified'] = False
            verification_result['errors'].append(f'Verification error: {e}')
            self.logger.error(f"Backup verification failed: {e}")
        
        return verification_result
    
    async def _cleanup_old_backups(self):
        """Clean up old backup files"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.backup_config['retention_days'])
            
            removed_backups = []
            total_space_freed = 0
            
            for backup_dir in self.backup_base_dir.iterdir():
                if backup_dir.is_dir():
                    # Check backup age from directory name or manifest
                    backup_age = None
                    
                    # Try to get age from manifest
                    manifest_path = backup_dir / 'backup_manifest.json'
                    if manifest_path.exists():
                        try:
                            with open(manifest_path, 'r') as f:
                                manifest = json.load(f)
                            backup_date = datetime.fromisoformat(manifest['start_time'])
                            backup_age = backup_date
                        except:
                            pass
                    
                    # Fallback to directory modification time
                    if backup_age is None:
                        backup_age = datetime.fromtimestamp(backup_dir.stat().st_mtime)
                    
                    if backup_age < cutoff_date:
                        # Calculate size before deletion
                        backup_size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
                        
                        # Remove old backup
                        shutil.rmtree(backup_dir)
                        
                        removed_backups.append(backup_dir.name)
                        total_space_freed += backup_size
                        
                        self.logger.info(f"Removed old backup: {backup_dir.name}")
            
            if removed_backups:
                self.logger.info(f"Cleanup completed: Removed {len(removed_backups)} old backups, "
                               f"freed {total_space_freed / (1024**3):.2f} GB")
            else:
                self.logger.info("No old backups to clean up")
                
        except Exception as e:
            self.logger.error(f"Error during backup cleanup: {e}")
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        
        backups = []
        
        try:
            for backup_dir in self.backup_base_dir.iterdir():
                if backup_dir.is_dir():
                    backup_info = {
                        'name': backup_dir.name,
                        'path': str(backup_dir),
                        'created': datetime.fromtimestamp(backup_dir.stat().st_ctime).isoformat(),
                        'size_bytes': sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file()),
                        'size_gb': 0
                    }
                    
                    backup_info['size_gb'] = round(backup_info['size_bytes'] / (1024**3), 2)
                    
                    # Try to get additional info from manifest
                    manifest_path = backup_dir / 'backup_manifest.json'
                    if manifest_path.exists():
                        try:
                            with open(manifest_path, 'r') as f:
                                manifest = json.load(f)
                            
                            backup_info.update({
                                'type': manifest.get('backup_type'),
                                'status': manifest.get('status'),
                                'components': manifest.get('components', []),
                                'compressed': manifest.get('compressed', False),
                                'verified': manifest.get('verification', {}).get('verified', False),
                                'start_time': manifest.get('start_time'),
                                'end_time': manifest.get('end_time')
                            })
                        except Exception as e:
                            backup_info['manifest_error'] = str(e)
                    
                    backups.append(backup_info)
            
            # Sort by creation date (newest first)
            backups.sort(key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error listing backups: {e}")
        
        return backups
    
    async def restore_backup(self, backup_name: str, components: List[str] = None, target_dir: str = None) -> Dict[str, Any]:
        """Restore from backup"""
        
        backup_dir = self.backup_base_dir / backup_name
        
        if not backup_dir.exists():
            raise Exception(f"Backup not found: {backup_name}")
        
        # Load backup manifest
        manifest_path = backup_dir / 'backup_manifest.json'
        if not manifest_path.exists():
            raise Exception(f"Backup manifest not found: {backup_name}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Determine components to restore
        available_components = manifest.get('components', [])
        if components is None:
            components_to_restore = available_components
        else:
            components_to_restore = [c for c in components if c in available_components]
        
        if not components_to_restore:
            raise Exception("No valid components specified for restoration")
        
        self.logger.info(f"Starting restoration of backup: {backup_name}")
        self.logger.info(f"Components to restore: {components_to_restore}")
        
        restore_result = {
            'backup_name': backup_name,
            'start_time': datetime.now().isoformat(),
            'components_restored': [],
            'components_failed': [],
            'status': 'in_progress',
            'target_directory': target_dir
        }
        
        try:
            for component in components_to_restore:
                self.logger.info(f"Restoring component: {component}")
                
                try:
                    await self._restore_component(component, backup_dir, target_dir)
                    restore_result['components_restored'].append(component)
                    self.logger.info(f"Successfully restored component: {component}")
                    
                except Exception as e:
                    restore_result['components_failed'].append({
                        'component': component,
                        'error': str(e)
                    })
                    self.logger.error(f"Failed to restore component {component}: {e}")
            
            restore_result['end_time'] = datetime.now().isoformat()
            
            if restore_result['components_failed']:
                restore_result['status'] = 'partial'
                self.logger.warning("Restoration completed with some failures")
            else:
                restore_result['status'] = 'completed'
                self.logger.info("Restoration completed successfully")
            
            return restore_result
            
        except Exception as e:
            restore_result['status'] = 'failed'
            restore_result['error'] = str(e)
            restore_result['end_time'] = datetime.now().isoformat()
            self.logger.error(f"Restoration failed: {e}")
            raise
    
    async def _restore_component(self, component: str, backup_dir: Path, target_dir: str = None):
        """Restore a specific component"""
        
        if component == 'database':
            await self._restore_database(backup_dir, target_dir)
        elif component == 'vectors':
            await self._restore_vectors(backup_dir, target_dir)
        elif component == 'documents':
            await self._restore_documents(backup_dir, target_dir)
        elif component == 'config':
            await self._restore_config(backup_dir, target_dir)
        elif component == 'models':
            await self._restore_models(backup_dir, target_dir)
        else:
            raise ValueError(f"Unknown component: {component}")
    
    async def _restore_database(self, backup_dir: Path, target_dir: str = None):
        """Restore PostgreSQL database"""
        
        # Find database backup file
        db_backup_file = None
        for file_path in backup_dir.iterdir():
            if file_path.name.startswith('postgresql_backup'):
                db_backup_file = file_path
                break
        
        if not db_backup_file:
            # Check in component subdirectory
            component_dir = backup_dir / 'database'
            if component_dir.exists():
                for file_path in component_dir.iterdir():
                    if file_path.name.startswith('postgresql_backup'):
                        db_backup_file = file_path
                        break
        
        if not db_backup_file:
            raise Exception("Database backup file not found")
        
        db_config = self.config.get('database', {})
        
        # Decompress if needed
        restore_file = db_backup_file
        if db_backup_file.name.endswith('.gz'):
            decompressed_file = backup_dir / 'postgresql_backup_temp.sql'
            with gzip.open(db_backup_file, 'rb') as f_in:
                with open(decompressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            restore_file = decompressed_file
        
        # Restore database
        cmd = [
            'psql',
            '-h', db_config.get('host', 'localhost'),
            '-p', str(db_config.get('port', 5432)),
            '-U', db_config.get('user', 'rag-system'),
            '-d', db_config.get('database', 'rag_metadata'),
            '--file', str(restore_file)
        ]
        
        # Set environment for password-less authentication
        env = os.environ.copy()
        env['PGPASSFILE'] = '/home/rag-system/.pgpass'
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        stdout, stderr = await process.communicate()
        
        # Cleanup temporary file
        if restore_file != db_backup_file and restore_file.exists():
            restore_file.unlink()
        
        if process.returncode != 0:
            raise Exception(f"Database restore failed: {stderr.decode()}")
    
    async def _restore_vectors(self, backup_dir: Path, target_dir: str = None):
        """Restore Qdrant vectors"""
        
        default_target = Path(self.config.get('data', {}).get('storage', {}).get('vectors_directory', '/data/rag/vectors'))
        vectors_target = Path(target_dir) / 'vectors' if target_dir else default_target
        
        # Find vectors backup
        vectors_backup = backup_dir / 'vectors'
        compressed_backup = None
        
        # Look for compressed backup
        for file_path in backup_dir.iterdir():
            if file_path.name.endswith('_vectors.tar.gz'):
                compressed_backup = file_path
                break
        
        if compressed_backup and compressed_backup.exists():
            # Extract compressed backup
            with tarfile.open(compressed_backup, 'r:gz') as tar:
                tar.extractall(path=vectors_target.parent)
                
        elif vectors_backup.exists():
            # Copy uncompressed backup
            if vectors_target.exists():
                shutil.rmtree(vectors_target)
            
            shutil.copytree(vectors_backup, vectors_target)
        else:
            raise Exception("Vectors backup not found")
    
    async def _restore_documents(self, backup_dir: Path, target_dir: str = None):
        """Restore documents"""
        
        default_target = Path(self.config.get('data', {}).get('storage', {}).get('documents_directory', '/data/rag/documents'))
        documents_target = Path(target_dir) / 'documents' if target_dir else default_target
        
        # Find documents backup
        documents_backup = backup_dir / 'documents'
        compressed_backup = None
        
        # Look for compressed backup
        for file_path in backup_dir.iterdir():
            if file_path.name.endswith('_documents.tar.gz'):
                compressed_backup = file_path
                break
        
        if compressed_backup and compressed_backup.exists():
            # Extract compressed backup
            with tarfile.open(compressed_backup, 'r:gz') as tar:
                tar.extractall(path=documents_target.parent)
                
        elif documents_backup.exists():
            # Copy uncompressed backup
            if documents_target.exists():
                shutil.rmtree(documents_target)
            
            shutil.copytree(documents_backup, documents_target)
        else:
            raise Exception("Documents backup not found")
    
    async def _restore_config(self, backup_dir: Path, target_dir: str = None):
        """Restore configuration files"""
        
        config_backup = backup_dir / 'config'
        compressed_backup = None
        
        # Look for compressed backup
        for file_path in backup_dir.iterdir():
            if file_path.name.endswith('_config.tar.gz'):
                compressed_backup = file_path
                break
        
        if compressed_backup and compressed_backup.exists():
            # Extract to temporary location first
            temp_dir = backup_dir / 'temp_config_restore'
            temp_dir.mkdir(exist_ok=True)
            
            with tarfile.open(compressed_backup, 'r:gz') as tar:
                tar.extractall(path=temp_dir)
            
            config_backup = temp_dir / 'config'
        
        if config_backup.exists():
            # Restore configuration files
            if target_dir:
                config_target = Path(target_dir) / 'config'
            else:
                config_target = Path('/opt/rag-system/config')
                
            if config_target.exists():
                # Backup existing config before restore
                backup_existing = config_target.parent / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(config_target, backup_existing)
                self.logger.info(f"Existing config backed up to: {backup_existing}")
            
            shutil.copytree(config_backup, config_target)
            
            # Clean up temp directory
            temp_dir = backup_dir / 'temp_config_restore'
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        else:
            raise Exception("Config backup not found")
    
    async def _restore_models(self, backup_dir: Path, target_dir: str = None):
        """Restore AI models"""
        
        compressed_backup = None
        
        # Look for compressed models backup
        for file_path in backup_dir.iterdir():
            if file_path.name.endswith('_models.tar.gz'):
                compressed_backup = file_path
                break
        
        if compressed_backup and compressed_backup.exists():
            # Extract models backup
            if target_dir:
                models_target = Path(target_dir)
            else:
                models_target = Path(self.config.get('data', {}).get('storage', {}).get('models_directory', '/opt/rag-system/models')).parent
            
            with tarfile.open(compressed_backup, 'r:gz') as tar:
                tar.extractall(path=models_target)
        else:
            raise Exception("Models backup not found")

    async def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics and health information"""
        
        try:
            backups = await self.list_backups()
            
            stats = {
                'total_backups': len(backups),
                'total_size_gb': sum(b['size_gb'] for b in backups),
                'by_type': {},
                'by_status': {},
                'oldest_backup': None,
                'newest_backup': None,
                'verified_backups': 0,
                'failed_backups': 0,
                'disk_usage': {}
            }
            
            if backups:
                stats['oldest_backup'] = min(backups, key=lambda x: x['created'])['created']
                stats['newest_backup'] = max(backups, key=lambda x: x['created'])['created']
                
                for backup in backups:
                    # Count by type
                    backup_type = backup.get('type', 'unknown')
                    stats['by_type'][backup_type] = stats['by_type'].get(backup_type, 0) + 1
                    
                    # Count by status
                    status = backup.get('status', 'unknown')
                    stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                    
                    # Count verified backups
                    if backup.get('verified'):
                        stats['verified_backups'] += 1
                    
                    # Count failed backups
                    if status == 'failed':
                        stats['failed_backups'] += 1
            
            # Get disk usage information
            try:
                import psutil
                disk_usage = psutil.disk_usage(str(self.backup_base_dir))
                stats['disk_usage'] = {
                    'total_gb': round(disk_usage.total / (1024**3), 2),
                    'used_gb': round(disk_usage.used / (1024**3), 2),
                    'free_gb': round(disk_usage.free / (1024**3), 2),
                    'backup_usage_percent': round((stats['total_size_gb'] / (disk_usage.total / (1024**3))) * 100, 2)
                }
            except:
                pass
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting backup statistics: {e}")
            return {}

async def main():
    """Main backup script entry point"""
    
    parser = argparse.ArgumentParser(description='RAG System Backup and Restore')
    parser.add_argument('action', choices=['backup', 'list', 'restore', 'stats', 'verify'], help='Action to perform')
    parser.add_argument('--type', choices=['full', 'data', 'config', 'database', 'minimal'], 
                       default='full', help='Backup type')
    parser.add_argument('--name', help='Custom backup name or backup to restore')
    parser.add_argument('--components', nargs='+', 
                       choices=['database', 'vectors', 'documents', 'config', 'models'],
                       help='Specific components to backup/restore')
    parser.add_argument('--target-dir', help='Target directory for restore (optional)')
    parser.add_argument('--config', help='Path to RAG config file', 
                       default='/opt/rag-system/config/rag_config.yaml')
    
    args = parser.parse_args()
    
    backup_system = RAGSystemBackup(config_path=args.config)
    
    try:
        if args.action == 'backup':
            result = await backup_system.create_backup(
                backup_type=args.type,
                custom_name=args.name
            )
            print(f"✅ Backup completed: {result['backup_name']}")
            print(f"📦 Size: {result['total_size_bytes'] / (1024**3):.2f} GB")
            print(f"🕒 Duration: {(datetime.fromisoformat(result['end_time']) - datetime.fromisoformat(result['start_time'])).total_seconds():.1f} seconds")
            
            if result.get('verification', {}).get('verified'):
                print("✅ Backup verified successfully")
            elif 'verification' in result:
                print("⚠️ Backup verification failed")
            
        elif args.action == 'list':
            backups = await backup_system.list_backups()
            
            if not backups:
                print("No backups found")
                return 0
            
            print(f"{'Name':<35} {'Type':<10} {'Size (GB)':<10} {'Created':<20} {'Status':<10} {'Verified'}")
            print("-" * 100)
            
            for backup in backups:
                name = backup['name'][:34]
                backup_type = backup.get('type', 'unknown')[:9]
                size_gb = backup['size_gb']
                created = backup['created'][:19].replace('T', ' ')
                status = backup.get('status', 'unknown')[:9]
                verified = "Yes" if backup.get('verified') else "No"
                
                print(f"{name:<35} {backup_type:<10} {size_gb:<10.2f} {created:<20} {status:<10} {verified}")
        
        elif args.action == 'restore':
            if not args.name:
                print("Error: --name is required for restore action")
                return 1
            
            result = await backup_system.restore_backup(
                backup_name=args.name,
                components=args.components,
                target_dir=args.target_dir
            )
            
            print(f"✅ Restoration completed: {result['status']}")
            if result['components_restored']:
                print(f"📦 Restored components: {', '.join(result['components_restored'])}")
            if result['components_failed']:
                print(f"❌ Failed components: {[c['component'] for c in result['components_failed']]}")
        
        elif args.action == 'stats':
            stats = await backup_system.get_backup_statistics()
            
            print("📊 Backup Statistics")
            print("=" * 50)
            print(f"Total backups: {stats.get('total_backups', 0)}")
            print(f"Total size: {stats.get('total_size_gb', 0):.2f} GB")
            print(f"Verified backups: {stats.get('verified_backups', 0)}")
            print(f"Failed backups: {stats.get('failed_backups', 0)}")
            
            if stats.get('newest_backup'):
                print(f"Newest backup: {stats['newest_backup'][:19]}")
            if stats.get('oldest_backup'):
                print(f"Oldest backup: {stats['oldest_backup'][:19]}")
            
            if stats.get('by_type'):
                print("\nBy type:")
                for backup_type, count in stats['by_type'].items():
                    print(f"  {backup_type}: {count}")
            
            if stats.get('disk_usage'):
                disk = stats['disk_usage']
                print(f"\nDisk usage:")
                print(f"  Total: {disk['total_gb']:.1f} GB")
                print(f"  Used: {disk['used_gb']:.1f} GB")
                print(f"  Free: {disk['free_gb']:.1f} GB")
                print(f"  Backup usage: {disk['backup_usage_percent']:.1f}%")
        
        elif args.action == 'verify':
            if not args.name:
                print("Error: --name is required for verify action")
                return 1
            
            backup_dir = backup_system.backup_base_dir / args.name
            if not backup_dir.exists():
                print(f"Error: Backup {args.name} not found")
                return 1
            
            verification_result = await backup_system._verify_backup(backup_dir)
            
            if verification_result['verified']:
                print(f"✅ Backup {args.name} verification PASSED")
            else:
                print(f"❌ Backup {args.name} verification FAILED")
            
            print(f"Checks passed: {len(verification_result['checks'])}")
            print(f"Errors found: {len(verification_result['errors'])}")
            
            if verification_result['errors']:
                print("\nErrors:")
                for error in verification_result['errors']:
                    print(f"  - {error}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))