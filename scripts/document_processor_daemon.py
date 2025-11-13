#!/usr/bin/env python3
"""
Document Processor Daemon for RAG System
Monitors directories for new files and processes them automatically
"""

import os
import sys
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import schedule

# Add RAG system to path
sys.path.append('/opt/rag-system')

from core.rag_pipeline import RAGPipeline
from core.document_processor import DocumentProcessor
from core.security_manager import SecurityManager

class DocumentProcessorDaemon:
    def __init__(self):
        self.logger = self._setup_logging()
        self.rag_pipeline = RAGPipeline()
        self.doc_processor = DocumentProcessor({})
        self.security_manager = SecurityManager({})
        
        # Configuration
        self.watch_directories = [
            '/data/rag/incoming/drivers',
            '/data/rag/incoming/embedded', 
            '/data/rag/incoming/radar',
            '/data/rag/incoming/general'
        ]
        
        self.processing_queue = asyncio.Queue()
        self.stats = {
            'processed': 0,
            'failed': 0,
            'start_time': datetime.now()
        }
        
        # Ensure directories exist
        self._ensure_directories()
        
        self.logger.info("Document Processor Daemon initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/rag/document_processor.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _ensure_directories(self):
        """Create watch directories if they don't exist"""
        for directory in self.watch_directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {directory}")
    
    async def start_daemon(self):
        """Start the document processing daemon"""
        self.logger.info("Starting Document Processor Daemon...")
        
        # Start file watcher
        event_handler = DocumentEventHandler(self)
        observer = Observer()
        
        for directory in self.watch_directories:
            observer.schedule(event_handler, directory, recursive=True)
            self.logger.info(f"Watching directory: {directory}")
        
        observer.start()
        
        # Start queue processor
        asyncio.create_task(self._process_queue())
        
        # Schedule periodic tasks
        schedule.every(1).hour.do(self._cleanup_processed_files)
        schedule.every(24).hours.do(self._generate_daily_report)
        
        try:
            while True:
                # Run scheduled tasks
                schedule.run_pending()
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            self.logger.info("Stopping Document Processor Daemon...")
            observer.stop()
        
        observer.join()
    
    async def queue_file_for_processing(self, file_path: str):
        """Add file to processing queue"""
        try:
            # Determine domain from path
            domain = self._determine_domain_from_path(file_path)
            
            # Determine security classification
            security_level = self._determine_security_from_path(file_path)
            
            # Create processing task
            task = {
                'file_path': file_path,
                'domain': domain,
                'security_classification': security_level,
                'queued_at': datetime.now().isoformat(),
                'attempts': 0
            }
            
            await self.processing_queue.put(task)
            self.logger.info(f"Queued file for processing: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error queuing file {file_path}: {e}")
    
    async def _process_queue(self):
        """Process files from the queue"""
        self.logger.info("Started queue processor")
        
        while True:
            try:
                # Get task from queue
                task = await self.processing_queue.get()
                
                # Process the file
                success = await self._process_file(task)
                
                if success:
                    self.stats['processed'] += 1
                    self.logger.info(f"Successfully processed: {task['file_path']}")
                    
                    # Move to processed directory
                    await self._move_to_processed(task['file_path'])
                    
                else:
                    self.stats['failed'] += 1
                    task['attempts'] += 1
                    
                    # Retry logic
                    if task['attempts'] < 3:
                        self.logger.warning(f"Retrying file processing: {task['file_path']} (attempt {task['attempts']})")
                        # Wait before retrying
                        await asyncio.sleep(60 * task['attempts'])
                        await self.processing_queue.put(task)
                    else:
                        self.logger.error(f"Failed to process after 3 attempts: {task['file_path']}")
                        await self._move_to_failed(task['file_path'])
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(5)
    
    async def _process_file(self, task: Dict) -> bool:
        """Process a single file"""
        try:
            file_path = task['file_path']
            
            # Validate file
            validation = await self.doc_processor.validate_file(file_path)
            if not validation['valid']:
                self.logger.error(f"File validation failed: {file_path} - {validation['issues']}")
                return False
            
            # Process with RAG pipeline
            success = await self.rag_pipeline.ingest_document(
                file_path=file_path,
                security_classification=task['security_classification'],
                domain=task['domain']
            )
            
            if success:
                self.logger.info(f"Document ingested successfully: {file_path}")
                return True
            else:
                self.logger.error(f"Document ingestion failed: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing file {task['file_path']}: {e}")
            return False
    
    def _determine_domain_from_path(self, file_path: str) -> str:
        """Determine domain based on file path"""
        path_lower = file_path.lower()
        
        if '/drivers/' in path_lower or '/driver/' in path_lower:
            return 'drivers'
        elif '/embedded/' in path_lower or '/firmware/' in path_lower:
            return 'embedded'
        elif '/radar/' in path_lower:
            return 'radar'
        elif '/rf/' in path_lower or '/radio/' in path_lower:
            return 'rf'
        elif '/ew/' in path_lower or '/electronic_warfare/' in path_lower:
            return 'ew'
        elif '/ate/' in path_lower or '/test/' in path_lower:
            return 'ate'
        else:
            return 'general'
    
    def _determine_security_from_path(self, file_path: str) -> str:
        """Determine security classification from file path"""
        path_lower = file_path.lower()
        
        if any(keyword in path_lower for keyword in ['classified', 'secret']):
            return 'classified'
        elif any(keyword in path_lower for keyword in ['confidential', 'restricted']):
            return 'confidential'
        elif any(keyword in path_lower for keyword in ['internal', 'proprietary']):
            return 'internal'
        else:
            return 'public'
    
    async def _move_to_processed(self, file_path: str):
        """Move processed file to processed directory"""
        try:
            processed_dir = '/data/rag/processed'
            Path(processed_dir).mkdir(parents=True, exist_ok=True)
            
            file_name = Path(file_path).name
            processed_path = Path(processed_dir) / file_name
            
            # If file exists, add timestamp
            if processed_path.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                stem = processed_path.stem
                suffix = processed_path.suffix
                processed_path = processed_path.parent / f"{stem}_{timestamp}{suffix}"
            
            Path(file_path).rename(processed_path)
            self.logger.info(f"Moved to processed: {file_path} -> {processed_path}")
            
        except Exception as e:
            self.logger.error(f"Error moving file to processed: {e}")
    
    async def _move_to_failed(self, file_path: str):
        """Move failed file to failed directory"""
        try:
            failed_dir = '/data/rag/failed'
            Path(failed_dir).mkdir(parents=True, exist_ok=True)
            
            file_name = Path(file_path).name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stem = Path(file_name).stem
            suffix = Path(file_name).suffix
            failed_path = Path(failed_dir) / f"{stem}_failed_{timestamp}{suffix}"
            
            Path(file_path).rename(failed_path)
            self.logger.error(f"Moved to failed: {file_path} -> {failed_path}")
            
        except Exception as e:
            self.logger.error(f"Error moving file to failed: {e}")
    
    def _cleanup_processed_files(self):
        """Clean up old processed files"""
        try:
            processed_dir = Path('/data/rag/processed')
            if processed_dir.exists():
                # Remove files older than 30 days
                cutoff_time = time.time() - (30 * 24 * 60 * 60)
                
                for file_path in processed_dir.iterdir():
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        self.logger.info(f"Cleaned up old processed file: {file_path}")
                        
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
    
    def _generate_daily_report(self):
        """Generate daily processing report"""
        try:
            uptime = datetime.now() - self.stats['start_time']
            
            report = {
                'date': datetime.now().isoformat(),
                'uptime_hours': uptime.total_seconds() / 3600,
                'files_processed': self.stats['processed'],
                'files_failed': self.stats['failed'],
                'success_rate': self.stats['processed'] / (self.stats['processed'] + self.stats['failed']) if (self.stats['processed'] + self.stats['failed']) > 0 else 0,
                'queue_size': self.processing_queue.qsize()
            }
            
            # Save report
            report_dir = Path('/var/log/rag/reports')
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Daily report generated: {report}")
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status"""
        uptime = datetime.now() - self.stats['start_time']
        
        return {
            'status': 'running',
            'uptime_seconds': uptime.total_seconds(),
            'files_processed': self.stats['processed'],
            'files_failed': self.stats['failed'],
            'queue_size': self.processing_queue.qsize(),
            'watch_directories': self.watch_directories,
            'success_rate': self.stats['processed'] / (self.stats['processed'] + self.stats['failed']) if (self.stats['processed'] + self.stats['failed']) > 0 else 1.0
        }

class DocumentEventHandler(FileSystemEventHandler):
    """Handle file system events for document processing"""
    
    def __init__(self, daemon: DocumentProcessorDaemon):
        self.daemon = daemon
        self.logger = daemon.logger
        
        # Supported file extensions
        self.supported_extensions = {'.c', '.cpp', '.h', '.hpp', '.py', '.m', 
                                   '.pdf', '.docx', '.doc', '.odt', '.txt', 
                                   '.md', '.rst', '.html', '.xml', '.json', 
                                   '.csv', '.xlsx', '.sql'}
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            self._handle_file_event(event.src_path, 'created')
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            self._handle_file_event(event.src_path, 'modified')
    
    def _handle_file_event(self, file_path: str, event_type: str):
        """Handle file events"""
        try:
            path = Path(file_path)
            
            # Check if file extension is supported
            if path.suffix.lower() not in self.supported_extensions:
                return
            
            # Skip temporary files
            if path.name.startswith('.') or path.name.startswith('~'):
                return
            
            # Wait a moment to ensure file is fully written
            time.sleep(2)
            
            # Check if file exists and is readable
            if not path.exists() or not path.is_file():
                return
            
            self.logger.info(f"File {event_type}: {file_path}")
            
            # Queue for processing
            asyncio.create_task(self.daemon.queue_file_for_processing(file_path))
            
        except Exception as e:
            self.logger.error(f"Error handling file event {file_path}: {e}")

async def main():
    """Main daemon entry point"""
    daemon = DocumentProcessorDaemon()
    
    try:
        await daemon.start_daemon()
    except KeyboardInterrupt:
        print("Document Processor Daemon stopped by user")
    except Exception as e:
        print(f"Document Processor Daemon crashed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))