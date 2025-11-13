"""
FastAPI service for RAG System
Provides REST API interface for document retrieval and query processing
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import traceback

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our RAG components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.rag_pipeline import RAGPipeline, RAGResponse
from core.security_manager import SecurityManager
from core.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/rag/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = Field(None, ge=1, le=50)
    include_sources: bool = True
    stream_response: bool = False

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[str]
    confidence_score: float
    processing_time: float
    retrieved_contexts: int
    timestamp: str

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    chunks_created: Optional[int] = None

class UserLoginRequest(BaseModel):
    user_id: str
    password: str  # In production, use proper authentication

class UserLoginResponse(BaseModel):
    success: bool
    session_token: str
    user_info: Dict[str, Any]
    expires_at: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]
    version: str = "1.0.0"

class StatsResponse(BaseModel):
    documents: Dict[str, Any]
    usage: Dict[str, Any]
    system: Dict[str, Any]

# FastAPI app
app = FastAPI(
    title="Data Patterns India RAG API",
    description="Retrieval-Augmented Generation API for defense electronics codebase and documentation",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances (will be initialized at startup)
rag_pipeline: Optional[RAGPipeline] = None
security_manager: Optional[SecurityManager] = None

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Validate user session and return user info"""
    try:
        if not security_manager:
            raise HTTPException(status_code=503, detail="Security manager not available")
        
        session_info = await security_manager.validate_session(credentials.credentials)
        if not session_info:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        
        return session_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=500, detail="Authentication service error")

# API Routes

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline and services"""
    global rag_pipeline, security_manager
    
    try:
        logger.info("Initializing RAG API services...")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        
        # Initialize security manager
        config = rag_pipeline.config  # Use same config
        security_manager = SecurityManager(config)
        
        # Perform health check
        health = await rag_pipeline.health_check()
        if health["status"] != "healthy":
            logger.warning(f"RAG pipeline health check shows issues: {health}")
        
        logger.info("RAG API services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG API: {e}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global rag_pipeline, security_manager
    
    try:
        if rag_pipeline:
            rag_pipeline.close()
        if security_manager:
            security_manager.close()
        logger.info("RAG API services shut down cleanly")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Data Patterns India RAG API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/api/docs"
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        health = await rag_pipeline.health_check()
        
        return HealthResponse(
            status=health["status"],
            timestamp=health["timestamp"],
            services=health.get("services", {}),
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="error",
            timestamp=datetime.now().isoformat(),
            services={"error": str(e)}
        )

@app.post("/api/auth/login", response_model=UserLoginResponse)
async def login(request: UserLoginRequest):
    """User login and session creation"""
    try:
        if not security_manager:
            raise HTTPException(status_code=503, detail="Security manager not available")
        
        # In production, implement proper password verification
        # For now, we'll create a session for any valid user_id
        user_clearance = await security_manager.get_user_clearance(request.user_id)
        
        # Create session
        session_token = await security_manager.create_session(request.user_id)
        
        return UserLoginResponse(
            success=True,
            session_token=session_token,
            user_info={
                "user_id": request.user_id,
                "security_clearance": user_clearance["security_clearance"],
                "domains": user_clearance["domains"]
            },
            expires_at=(datetime.now().timestamp() + 8*3600)  # 8 hours
        )
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/auth/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """User logout"""
    try:
        if not security_manager:
            raise HTTPException(status_code=503, detail="Security manager not available")
        
        await security_manager.logout_session(current_user["session_id"])
        return {"success": True, "message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Main RAG query endpoint"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        # Process query
        response = await rag_pipeline.query(
            query_text=request.query,
            user_id=current_user["user_id"],
            filters=request.filters,
            top_k=request.top_k
        )
        
        # Check for security violations
        if security_manager:
            violations = await security_manager.detect_security_violations(
                current_user["user_id"],
                request.query,
                response.sources
            )
            
            if violations:
                high_severity_violations = [v for v in violations if v["severity"] == "high"]
                if high_severity_violations:
                    raise HTTPException(
                        status_code=403,
                        detail="Query blocked due to security policy violations"
                    )
        
        return QueryResponse(
            query=response.query,
            response=response.generated_response,
            sources=response.sources if request.include_sources else [],
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            retrieved_contexts=len(response.retrieved_contexts),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Query processing failed")

@app.post("/api/query/stream")
async def stream_query_rag(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Streaming RAG query endpoint"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        async def generate_stream():
            try:
                # Get response
                response = await rag_pipeline.query(
                    query_text=request.query,
                    user_id=current_user["user_id"],
                    filters=request.filters,
                    top_k=request.top_k
                )
                
                # Stream the response word by word
                words = response.generated_response.split()
                for i, word in enumerate(words):
                    chunk = {
                        "type": "content",
                        "content": word + " ",
                        "index": i
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.05)  # Small delay for streaming effect
                
                # Send final metadata
                final_chunk = {
                    "type": "metadata",
                    "sources": response.sources,
                    "confidence_score": response.confidence_score,
                    "processing_time": response.processing_time
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_chunk = {
                    "type": "error",
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming query error: {e}")
        raise HTTPException(status_code=500, detail="Streaming query failed")

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    security_classification: str = Form("internal"),
    domain: str = Form("general"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Upload and process a document"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        # Check if user can upload to this security level
        user_clearance = await security_manager.get_user_clearance(current_user["user_id"])
        user_level = security_manager.security_levels.get(user_clearance["security_clearance"], 0)
        doc_level = security_manager.security_levels.get(security_classification, 4)
        
        if user_level < doc_level:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient clearance to upload {security_classification} documents"
            )
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process the document
            success = await rag_pipeline.ingest_document(
                tmp_file_path,
                security_classification,
                domain
            )
            
            if success:
                return DocumentUploadResponse(
                    success=True,
                    message="Document uploaded and processed successfully",
                    document_id=f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    chunks_created=None  # Could be enhanced to return actual count
                )
            else:
                raise HTTPException(status_code=422, detail="Document processing failed")
                
        finally:
            # Clean up temporary file
            Path(tmp_file_path).unlink()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail="Document upload failed")

@app.post("/api/documents/batch-ingest")
async def batch_ingest_documents(
    directory_path: str = Form(...),
    recursive: bool = Form(True),
    file_patterns: Optional[List[str]] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Batch ingest documents from a directory"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        # Check if user has sufficient privileges for batch operations
        user_clearance = await security_manager.get_user_clearance(current_user["user_id"])
        if user_clearance["security_clearance"] not in ["confidential", "classified"]:
            raise HTTPException(
                status_code=403,
                detail="Batch ingestion requires elevated privileges"
            )
        
        # Start batch ingestion
        results = await rag_pipeline.batch_ingest(
            directory_path,
            recursive,
            file_patterns
        )
        
        return {
            "success": True,
            "message": "Batch ingestion completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch ingestion error: {e}")
        raise HTTPException(status_code=500, detail="Batch ingestion failed")

@app.get("/api/documents/stats", response_model=StatsResponse)
async def get_document_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get document and system statistics"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        stats = rag_pipeline.get_stats()
        
        return StatsResponse(
            documents=stats.get("documents", {}),
            usage=stats.get("usage", {}),
            system=stats.get("system", {})
        )
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.get("/api/security/dashboard")
async def get_security_dashboard(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get security dashboard data (admin only)"""
    try:
        if not security_manager:
            raise HTTPException(status_code=503, detail="Security manager not available")
        
        # Check if user has admin privileges
        user_clearance = await security_manager.get_user_clearance(current_user["user_id"])
        if user_clearance["security_clearance"] not in ["confidential", "classified"]:
            raise HTTPException(
                status_code=403,
                detail="Security dashboard requires elevated privileges"
            )
        
        dashboard_data = await security_manager.get_security_dashboard()
        return dashboard_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security dashboard error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security dashboard")

@app.get("/api/security/user-activity/{user_id}")
async def get_user_activity(
    user_id: str,
    days: int = Query(7, ge=1, le=30),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get user activity summary"""
    try:
        if not security_manager:
            raise HTTPException(status_code=503, detail="Security manager not available")
        
        # Users can only see their own activity unless they have admin privileges
        user_clearance = await security_manager.get_user_clearance(current_user["user_id"])
        if (user_id != current_user["user_id"] and 
            user_clearance["security_clearance"] not in ["confidential", "classified"]):
            raise HTTPException(
                status_code=403,
                detail="Cannot access other users' activity"
            )
        
        activity = await security_manager.get_user_activity_summary(user_id, days)
        return activity
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User activity error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user activity")

@app.post("/api/admin/user-clearance")
async def set_user_clearance(
    user_id: str = Form(...),
    security_level: str = Form(...),
    domains: List[str] = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Set user security clearance (admin only)"""
    try:
        if not security_manager:
            raise HTTPException(status_code=503, detail="Security manager not available")
        
        # Check if current user has admin privileges
        user_clearance = await security_manager.get_user_clearance(current_user["user_id"])
        if user_clearance["security_clearance"] != "classified":
            raise HTTPException(
                status_code=403,
                detail="Setting user clearance requires classified access"
            )
        
        success = await security_manager.set_user_clearance(
            user_id,
            security_level,
            domains,
            current_user["user_id"]
        )
        
        if success:
            return {
                "success": True,
                "message": f"User {user_id} clearance set to {security_level}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to set user clearance")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Set user clearance error: {e}")
        raise HTTPException(status_code=500, detail="Failed to set user clearance")

@app.get("/api/documents/search")
async def search_documents(
    query: str = Query(..., min_length=1),
    domain_filter: Optional[str] = Query(None),
    file_type_filter: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Search documents metadata (without full content)"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        # Build search filters
        filters = {}
        if domain_filter:
            filters["domain"] = domain_filter
        if file_type_filter:
            filters["file_type"] = file_type_filter
        
        # This would need to be implemented in the RAG pipeline
        # For now, return a placeholder response
        return {
            "query": query,
            "filters": filters,
            "results": [],
            "total_count": 0,
            "message": "Document search endpoint - implementation needed"
        }
        
    except Exception as e:
        logger.error(f"Document search error: {e}")
        raise HTTPException(status_code=500, detail="Document search failed")

@app.get("/api/documents/{document_id}")
async def get_document_info(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get document information and access control"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        # This would need to be implemented in the RAG pipeline
        # For now, return a placeholder response
        return {
            "document_id": document_id,
            "message": "Document info endpoint - implementation needed",
            "access_granted": True
        }
        
    except Exception as e:
        logger.error(f"Get document info error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document info")

@app.delete("/api/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a document (admin only)"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        # Check if user has delete privileges
        user_clearance = await security_manager.get_user_clearance(current_user["user_id"])
        if user_clearance["security_clearance"] not in ["confidential", "classified"]:
            raise HTTPException(
                status_code=403,
                detail="Document deletion requires elevated privileges"
            )
        
        # This would need to be implemented in the RAG pipeline
        # For now, return a placeholder response
        return {
            "success": True,
            "message": f"Document {document_id} deletion requested",
            "note": "Delete endpoint - implementation needed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.post("/api/maintenance/cleanup")
async def cleanup_old_data(
    days_old: int = Form(30),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Clean up old data and logs (admin only)"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        # Check if user has admin privileges
        user_clearance = await security_manager.get_user_clearance(current_user["user_id"])
        if user_clearance["security_clearance"] != "classified":
            raise HTTPException(
                status_code=403,
                detail="Maintenance operations require classified access"
            )
        
        # Cleanup old data
        await rag_pipeline.cleanup_old_data(days_old)
        
        # Cleanup expired sessions
        if security_manager:
            await security_manager.cleanup_expired_sessions()
        
        return {
            "success": True,
            "message": f"Cleaned up data older than {days_old} days",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail="Cleanup operation failed")

@app.get("/api/models/status")
async def get_models_status(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get status of AI models"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
        health = await rag_pipeline.health_check()
        
        return {
            "models": health.get("models", {}),
            "embedding_models": list(rag_pipeline.embedding_models.keys()) if rag_pipeline else [],
            "vector_collections": health.get("data", {}).get("collections", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Models status error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get models status")

# WebSocket endpoint for real-time features
@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket, user_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    try:
        # Validate user session (would need to implement WebSocket auth)
        logger.info(f"WebSocket connection established for user: {user_id}")
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "query":
                # Process query via RAG pipeline
                try:
                    if rag_pipeline:
                        response = await rag_pipeline.query(
                            query_text=message["query"],
                            user_id=user_id
                        )
                        
                        await websocket.send_text(json.dumps({
                            "type": "response",
                            "response": response.generated_response,
                            "sources": response.sources,
                            "confidence": response.confidence_score,
                            "timestamp": datetime.now().isoformat()
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "RAG pipeline not available"
                        }))
                        
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Query processing error: {str(e)}"
                    }))
            
            elif message["type"] == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
    finally:
        logger.info(f"WebSocket connection closed for user: {user_id}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server configuration
if __name__ == "__main__":
    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        workers=1  # Single worker for development
    )