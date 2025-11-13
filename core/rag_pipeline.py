"""
Core RAG Pipeline for Data Patterns India
Integrates with existing DeepSeek pipeline and provides contextual retrieval
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import psycopg2
import redis
from transformers import AutoTokenizer

from .document_processor import DocumentProcessor
from .security_manager import SecurityManager
from .query_optimizer import QueryOptimizer

@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk"""
    content: str
    source_file: str
    content_type: str
    score: float
    metadata: Dict[str, Any]
    security_level: str

@dataclass
class RAGResponse:
    """Complete RAG response with context and generation"""
    query: str
    retrieved_contexts: List[RetrievalResult]
    generated_response: str
    confidence_score: float
    sources: List[str]
    processing_time: float

class RAGPipeline:
    """Main RAG pipeline for code and document retrieval"""
    
    def __init__(self, config_path: str = "/opt/rag-system/config/rag_config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.embedding_models = self._load_embedding_models()
        self.qdrant_client = self._init_qdrant()
        self.db_connection = self._init_database()
        self.redis_client = self._init_redis()
        self.security_manager = SecurityManager(self.config)
        self.query_optimizer = QueryOptimizer(self.config)
        self.document_processor = DocumentProcessor(self.config)
        
        # Initialize collections if they don't exist
        self._ensure_collections()
        
        self.logger.info("RAG Pipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                "embedding": {
                    "primary_model": "bge-m3",
                    "code_model": "codebert-base",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "max_chunks_per_doc": 50
                },
                "retrieval": {
                    "top_k": 10,
                    "score_threshold": 0.7,
                    "rerank_top_k": 5,
                    "hybrid_search_weight": 0.7
                },
                "security": {
                    "classification_levels": ["public", "internal", "confidential", "classified"],
                    "audit_all_queries": True
                },
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
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/rag/rag_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_embedding_models(self) -> Dict[str, SentenceTransformer]:
        """Load pre-trained embedding models"""
        models = {}
        model_dir = "/opt/rag-system/models"
        
        try:
            # Primary embedding model (BGE-M3)
            models['primary'] = SentenceTransformer(f"{model_dir}/bge-m3")
            
            # Code-specific model (CodeBERT)
            models['code'] = SentenceTransformer(f"{model_dir}/codebert-base")
            
            # General model (E5-large-v2)
            models['general'] = SentenceTransformer(f"{model_dir}/e5-large-v2")
            
            self.logger.info(f"Loaded {len(models)} embedding models")
            return models
            
        except Exception as e:
            self.logger.error(f"Error loading embedding models: {e}")
            raise
    
    def _init_qdrant(self) -> QdrantClient:
        """Initialize Qdrant vector database client"""
        try:
            client = QdrantClient(
                host=self.config["qdrant"]["host"],
                port=self.config["qdrant"]["port"]
            )
            self.logger.info("Connected to Qdrant vector database")
            return client
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _init_database(self) -> psycopg2.extensions.connection:
        """Initialize PostgreSQL connection"""
        try:
            conn = psycopg2.connect(
                host=self.config["database"]["host"],
                port=self.config["database"]["port"],
                database=self.config["database"]["database"],
                user=self.config["database"]["user"]
            )
            self.logger.info("Connected to PostgreSQL database")
            return conn
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis cache client"""
        try:
            client = redis.Redis(
                host=self.config["redis"]["host"],
                port=self.config["redis"]["port"],
                db=self.config["redis"]["db"],
                decode_responses=True
            )
            # Test connection
            client.ping()
            self.logger.info("Connected to Redis cache")
            return client
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _ensure_collections(self):
        """Ensure required Qdrant collections exist"""
        collections = [
            ("documents", 1024),  # BGE-M3 dimension
            ("code_chunks", 768),  # CodeBERT dimension
            ("hybrid_search", 1024)  # Primary model for hybrid search
        ]
        
        for collection_name, vector_size in collections:
            try:
                # Check if collection exists
                collections_info = self.qdrant_client.get_collections()
                existing_names = [c.name for c in collections_info.collections]
                
                if collection_name not in existing_names:
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE
                        )
                    )
                    self.logger.info(f"Created Qdrant collection: {collection_name}")
                else:
                    self.logger.info(f"Collection {collection_name} already exists")
                    
            except Exception as e:
                self.logger.error(f"Error ensuring collection {collection_name}: {e}")
                raise
    
    async def query(self, 
                   query_text: str, 
                   user_id: str,
                   filters: Optional[Dict[str, Any]] = None,
                   top_k: int = None) -> RAGResponse:
        """
        Main query interface for RAG system
        
        Args:
            query_text: User's query
            user_id: User identifier for security and auditing
            filters: Optional filters (domain, file_type, security_level)
            top_k: Number of results to return
            
        Returns:
            RAGResponse with retrieved context and generated answer
        """
        start_time = datetime.now()
        
        try:
            # Security check and audit logging
            user_clearance = await self.security_manager.get_user_clearance(user_id)
            await self._audit_query(user_id, query_text)
            
            # Optimize query
            optimized_query = await self.query_optimizer.optimize(query_text)
            
            # Check cache first
            cache_key = self._generate_cache_key(query_text, filters, user_clearance)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"Returning cached result for user {user_id}")
                return cached_result
            
            # Retrieve relevant contexts
            retrieved_contexts = await self._retrieve_contexts(
                optimized_query, 
                user_clearance, 
                filters, 
                top_k or self.config["retrieval"]["top_k"]
            )
            
            # Generate response using DeepSeek
            generated_response = await self._generate_response(
                query_text, 
                retrieved_contexts
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(retrieved_contexts, generated_response)
            
            # Create response object
            response = RAGResponse(
                query=query_text,
                retrieved_contexts=retrieved_contexts,
                generated_response=generated_response,
                confidence_score=confidence_score,
                sources=list(set([ctx.source_file for ctx in retrieved_contexts])),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Cache the result
            await self._cache_result(cache_key, response)
            
            # Log successful query
            await self._log_successful_query(user_id, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query for user {user_id}: {e}")
            raise
    
    async def _retrieve_contexts(self, 
                               query: str, 
                               user_clearance: Dict[str, Any],
                               filters: Optional[Dict[str, Any]],
                               top_k: int) -> List[RetrievalResult]:
        """Retrieve relevant contexts using hybrid search"""
        
        # Generate embeddings for the query
        query_embedding = await self._generate_query_embedding(query)
        
        # Determine search strategy based on query type
        search_strategy = self._determine_search_strategy(query)
        
        # Perform vector search
        if search_strategy == "code":
            collection_name = "code_chunks"
            model_key = "code"
        elif search_strategy == "hybrid":
            collection_name = "hybrid_search"
            model_key = "primary"
        else:
            collection_name = "documents"
            model_key = "primary"
        
        # Build Qdrant filter
        qdrant_filter = self._build_qdrant_filter(user_clearance, filters)
        
        # Search vectors
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding[model_key],
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=self.config["retrieval"]["score_threshold"]
        )
        
        # Convert to RetrievalResult objects
        contexts = []
        for result in search_results:
            try:
                # Get full content from database
                content_data = await self._get_chunk_content(result.id)
                if content_data:
                    contexts.append(RetrievalResult(
                        content=content_data["content"],
                        source_file=content_data["source_file"],
                        content_type=content_data["content_type"],
                        score=result.score,
                        metadata=content_data["metadata"],
                        security_level=content_data["security_level"]
                    ))
            except Exception as e:
                self.logger.warning(f"Error retrieving chunk {result.id}: {e}")
                continue
        
        # Re-rank results if needed
        if len(contexts) > self.config["retrieval"]["rerank_top_k"]:
            contexts = await self._rerank_contexts(query, contexts)
        
        return contexts[:self.config["retrieval"]["rerank_top_k"]]
    
    async def _generate_query_embedding(self, query: str) -> Dict[str, np.ndarray]:
        """Generate embeddings for query using multiple models"""
        embeddings = {}
        
        try:
            # Primary model embedding
            embeddings["primary"] = self.embedding_models["primary"].encode(
                query, normalize_embeddings=True
            )
            
            # Code model embedding (if query seems code-related)
            if self._is_code_query(query):
                embeddings["code"] = self.embedding_models["code"].encode(
                    query, normalize_embeddings=True
                )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating query embeddings: {e}")
            raise
    
    def _determine_search_strategy(self, query: str) -> str:
        """Determine the best search strategy based on query content"""
        code_keywords = [
            "function", "class", "method", "variable", "implement", "code",
            "algorithm", "bug", "error", "debug", "compile", "syntax"
        ]
        
        doc_keywords = [
            "document", "specification", "requirement", "design", "manual",
            "guide", "procedure", "policy", "standard"
        ]
        
        query_lower = query.lower()
        
        code_score = sum(1 for keyword in code_keywords if keyword in query_lower)
        doc_score = sum(1 for keyword in doc_keywords if keyword in query_lower)
        
        if code_score > doc_score and code_score > 2:
            return "code"
        elif doc_score > code_score and doc_score > 2:
            return "documents"
        else:
            return "hybrid"
    
    def _is_code_query(self, query: str) -> bool:
        """Determine if query is code-related"""
        code_indicators = [
            "function", "class", "method", "variable", "implement", "code",
            "algorithm", "bug", "error", "debug", "#include", "import",
            "def ", "void ", "int ", "class ", "struct", "typedef"
        ]
        return any(indicator in query.lower() for indicator in code_indicators)
    
    def _build_qdrant_filter(self, 
                           user_clearance: Dict[str, Any], 
                           filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build Qdrant filter based on security clearance and user filters"""
        qdrant_filter = {
            "must": []
        }
        
        # Security filter - user can only access documents at or below their clearance
        allowed_levels = self.security_manager.get_allowed_security_levels(
            user_clearance["security_clearance"]
        )
        qdrant_filter["must"].append({
            "key": "security_level",
            "match": {"any": allowed_levels}
        })
        
        # Domain filter
        if user_clearance.get("domains"):
            qdrant_filter["must"].append({
                "key": "domain",
                "match": {"any": user_clearance["domains"]}
            })
        
        # Additional user-specified filters
        if filters:
            if "domain" in filters:
                qdrant_filter["must"].append({
                    "key": "domain",
                    "match": {"value": filters["domain"]}
                })
            
            if "file_type" in filters:
                qdrant_filter["must"].append({
                    "key": "file_type",
                    "match": {"value": filters["file_type"]}
                })
            
            if "language" in filters:
                qdrant_filter["must"].append({
                    "key": "language",
                    "match": {"value": filters["language"]}
                })
        
        return qdrant_filter
    
    async def _get_chunk_content(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full chunk content from database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT c.content, c.content_type, c.chunk_metadata,
                       d.file_path, d.security_classification, d.domain, d.language
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding_id = %s
            """, (chunk_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    "content": result[0],
                    "content_type": result[1],
                    "metadata": result[2] or {},
                    "source_file": result[3],
                    "security_level": result[4],
                    "domain": result[5],
                    "language": result[6]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving chunk content: {e}")
            return None
    
    async def _rerank_contexts(self, 
                             query: str, 
                             contexts: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank retrieved contexts for better relevance"""
        try:
            # Simple re-ranking based on query-context similarity
            query_embedding = self.embedding_models["primary"].encode(query)
            
            for context in contexts:
                context_embedding = self.embedding_models["primary"].encode(context.content[:500])
                # Update score with semantic similarity
                similarity = np.dot(query_embedding, context_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
                )
                context.score = 0.7 * context.score + 0.3 * similarity
            
            # Sort by updated scores
            contexts.sort(key=lambda x: x.score, reverse=True)
            return contexts
            
        except Exception as e:
            self.logger.warning(f"Error re-ranking contexts: {e}")
            return contexts
    
    async def _generate_response(self, 
                               query: str, 
                               contexts: List[RetrievalResult]) -> str:
        """Generate response using DeepSeek with retrieved context"""
        try:
            # Prepare context for DeepSeek
            context_text = self._format_contexts_for_llm(contexts)
            
            # Create prompt
            system_prompt = """You are an expert software engineer and technical documentation assistant for Data Patterns India, a defense electronics company. You have access to the company's codebase, documentation, and technical reports.

Your role is to provide accurate, helpful, and secure responses based on the retrieved context. When answering:

1. Base your response primarily on the provided context
2. Be specific and cite relevant code examples or documentation
3. Respect security classifications - don't reveal sensitive information beyond what's in the context
4. If the context is insufficient, clearly state what information is missing
5. Provide practical, actionable guidance
6. Use appropriate technical terminology for the domain (drivers, embedded, radar, etc.)

Always maintain professionalism and accuracy in your responses."""

            user_prompt = f"""Based on the following context from Data Patterns India's technical documentation and codebase, please answer the user's question.

CONTEXT:
{context_text}

USER QUESTION: {query}

Please provide a comprehensive answer based on the context provided. If you need additional information that's not in the context, please specify what would be helpful."""

            # Use existing DeepSeek client if available
            if hasattr(self, 'deepseek_client'):
                response = self.deepseek_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=2000
                )
                return response.content.strip()
            else:
                # Fallback to OpenAI API format (for OpenWebUI integration)
                import requests
                api_response = requests.post(
                    "http://localhost:11434/v1/chat/completions",
                    json={
                        "model": "deepseek-r1:70b",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 2000
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if api_response.status_code == 200:
                    return api_response.json()["choices"][0]["message"]["content"].strip()
                else:
                    raise Exception(f"DeepSeek API error: {api_response.status_code}")
                    
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response. The retrieved context suggests information about {', '.join([ctx.source_file for ctx in contexts[:3]])}, but I cannot provide a complete answer at this time."
    
    def _format_contexts_for_llm(self, contexts: List[RetrievalResult]) -> str:
        """Format retrieved contexts for LLM consumption"""
        formatted_contexts = []
        
        for i, context in enumerate(contexts, 1):
            formatted_context = f"""
--- Context {i} ---
Source: {context.source_file}
Type: {context.content_type}
Relevance Score: {context.score:.3f}

Content:
{context.content}

Metadata: {json.dumps(context.metadata, indent=2)}
---
"""
            formatted_contexts.append(formatted_context)
        
        return "\n".join(formatted_contexts)
    
    def _calculate_confidence(self, 
                            contexts: List[RetrievalResult], 
                            response: str) -> float:
        """Calculate confidence score for the response"""
        if not contexts:
            return 0.0
        
        # Base confidence on retrieval scores
        avg_retrieval_score = sum(ctx.score for ctx in contexts) / len(contexts)
        
        # Factor in number of high-quality contexts
        high_quality_contexts = sum(1 for ctx in contexts if ctx.score > 0.8)
        quality_factor = min(high_quality_contexts / 3, 1.0)  # Normalize to max 1.0
        
        # Factor in response length (reasonable responses should have substance)
        length_factor = min(len(response.split()) / 100, 1.0)  # Normalize to max 1.0
        
        # Combine factors
        confidence = (
            0.5 * avg_retrieval_score +
            0.3 * quality_factor +
            0.2 * length_factor
        )
        
        return min(confidence, 1.0)
    
    def _generate_cache_key(self, 
                          query: str, 
                          filters: Optional[Dict[str, Any]], 
                          user_clearance: Dict[str, Any]) -> str:
        """Generate cache key for query"""
        cache_data = {
            "query": query,
            "filters": filters or {},
            "security_level": user_clearance["security_clearance"],
            "domains": sorted(user_clearance.get("domains", []))
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[RAGResponse]:
        """Retrieve cached result if available"""
        try:
            cached_data = self.redis_client.get(f"rag_cache:{cache_key}")
            if cached_data:
                return RAGResponse(**json.loads(cached_data))
            return None
        except Exception as e:
            self.logger.warning(f"Error retrieving cached result: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, response: RAGResponse):
        """Cache the response for future queries"""
        try:
            cache_data = {
                "query": response.query,
                "retrieved_contexts": [
                    {
                        "content": ctx.content,
                        "source_file": ctx.source_file,
                        "content_type": ctx.content_type,
                        "score": ctx.score,
                        "metadata": ctx.metadata,
                        "security_level": ctx.security_level
                    }
                    for ctx in response.retrieved_contexts
                ],
                "generated_response": response.generated_response,
                "confidence_score": response.confidence_score,
                "sources": response.sources,
                "processing_time": response.processing_time
            }
            
            # Cache for 1 hour
            self.redis_client.setex(
                f"rag_cache:{cache_key}",
                3600,
                json.dumps(cache_data)
            )
        except Exception as e:
            self.logger.warning(f"Error caching result: {e}")
    
    async def _audit_query(self, user_id: str, query: str):
        """Log query for audit purposes"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO query_log (user_id, query, timestamp)
                VALUES (%s, %s, %s)
            """, (user_id, query, datetime.now()))
            self.db_connection.commit()
        except Exception as e:
            self.logger.error(f"Error logging query audit: {e}")
    
    async def _log_successful_query(self, user_id: str, response: RAGResponse):
        """Log successful query with retrieved documents"""
        try:
            # Get document IDs for retrieved contexts
            doc_ids = []
            cursor = self.db_connection.cursor()
            
            for source_file in response.sources:
                cursor.execute(
                    "SELECT id FROM documents WHERE file_path = %s",
                    (source_file,)
                )
                result = cursor.fetchone()
                if result:
                    doc_ids.append(str(result[0]))
            
            # Update the latest query log entry
            cursor.execute("""
                UPDATE query_log 
                SET retrieved_docs = %s, response_generated = true,
                    metadata = %s
                WHERE user_id = %s AND query = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (
                doc_ids,
                json.dumps({
                    "confidence_score": response.confidence_score,
                    "processing_time": response.processing_time,
                    "num_contexts": len(response.retrieved_contexts)
                }),
                user_id,
                response.query
            ))
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error logging successful query: {e}")
    
    # Document ingestion methods
    async def ingest_document(self, 
                            file_path: str, 
                            security_classification: str = "internal",
                            domain: str = "general") -> bool:
        """Ingest a single document into the RAG system"""
        try:
            # Check if document already exists
            if await self._document_exists(file_path):
                self.logger.info(f"Document {file_path} already exists, skipping")
                return True
            
            # Process document
            processed_doc = await self.document_processor.process_file(
                file_path, security_classification, domain
            )
            
            if not processed_doc:
                self.logger.error(f"Failed to process document: {file_path}")
                return False
            
            # Store in database
            doc_id = await self._store_document_metadata(processed_doc)
            
            # Generate embeddings and store in vector database
            await self._store_document_embeddings(doc_id, processed_doc)
            
            self.logger.info(f"Successfully ingested document: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ingesting document {file_path}: {e}")
            return False
    
    async def _document_exists(self, file_path: str) -> bool:
        """Check if document already exists in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT id FROM documents WHERE file_path = %s",
                (file_path,)
            )
            return cursor.fetchone() is not None
        except Exception as e:
            self.logger.error(f"Error checking document existence: {e}")
            return False
    
    async def _store_document_metadata(self, processed_doc: Dict[str, Any]) -> str:
        """Store document metadata in PostgreSQL"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO documents (
                    file_path, file_name, file_type, file_size, content_hash,
                    security_classification, domain, language, metadata, last_processed
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                processed_doc["file_path"],
                processed_doc["file_name"],
                processed_doc["file_type"],
                processed_doc["file_size"],
                processed_doc["content_hash"],
                processed_doc["security_classification"],
                processed_doc["domain"],
                processed_doc["language"],
                json.dumps(processed_doc["metadata"]),
                datetime.now()
            ))
            
            doc_id = cursor.fetchone()[0]
            self.db_connection.commit()
            return str(doc_id)
            
        except Exception as e:
            self.logger.error(f"Error storing document metadata: {e}")
            raise
    
    async def _store_document_embeddings(self, doc_id: str, processed_doc: Dict[str, Any]):
        """Generate and store document embeddings in Qdrant"""
        try:
            for i, chunk in enumerate(processed_doc["chunks"]):
                # Generate embeddings using appropriate model
                if chunk["content_type"] == "code":
                    embedding = self.embedding_models["code"].encode(
                        chunk["content"], normalize_embeddings=True
                    )
                    collection_name = "code_chunks"
                else:
                    embedding = self.embedding_models["primary"].encode(
                        chunk["content"], normalize_embeddings=True
                    )
                    collection_name = "documents"
                
                # Create unique ID for the chunk
                chunk_id = f"{doc_id}_{i}"
                
                # Store in PostgreSQL
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT INTO chunks (
                        document_id, chunk_index, content, content_type,
                        chunk_metadata, embedding_id
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    doc_id, i, chunk["content"], chunk["content_type"],
                    json.dumps(chunk.get("metadata", {})), chunk_id
                ))
                
                # Store in Qdrant
                point = PointStruct(
                    id=chunk_id,
                    vector=embedding.tolist(),
                    payload={
                        "document_id": doc_id,
                        "chunk_index": i,
                        "content_type": chunk["content_type"],
                        "security_level": processed_doc["security_classification"],
                        "domain": processed_doc["domain"],
                        "language": processed_doc["language"],
                        "file_type": processed_doc["file_type"],
                        "file_path": processed_doc["file_path"]
                    }
                )
                
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[point]
                )
            
            self.db_connection.commit()
            self.logger.info(f"Stored {len(processed_doc['chunks'])} chunks for document {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing document embeddings: {e}")
            raise
    
    async def batch_ingest(self, 
                         directory_path: str, 
                         recursive: bool = True,
                         file_patterns: List[str] = None) -> Dict[str, int]:
        """Batch ingest documents from a directory"""
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        try:
            # Get all files to process
            files_to_process = self.document_processor.find_files(
                directory_path, recursive, file_patterns
            )
            
            self.logger.info(f"Found {len(files_to_process)} files to process")
            
            # Process files in batches
            batch_size = 10
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i+batch_size]
                
                # Process batch concurrently
                tasks = []
                for file_path in batch:
                    # Determine domain and security level from path
                    domain = self._determine_domain_from_path(file_path)
                    security_level = self._determine_security_from_path(file_path)
                    
                    task = self.ingest_document(file_path, security_level, domain)
                    tasks.append(task)
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count results
                for result in batch_results:
                    if isinstance(result, Exception):
                        results["failed"] += 1
                    elif result:
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                
                self.logger.info(f"Processed batch {i//batch_size + 1}, "
                               f"Success: {results['success']}, "
                               f"Failed: {results['failed']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch ingestion: {e}")
            raise
    
    def _determine_domain_from_path(self, file_path: str) -> str:
        """Determine domain from file path"""
        path_lower = file_path.lower()
        
        if "driver" in path_lower or "hal" in path_lower:
            return "drivers"
        elif "embedded" in path_lower or "firmware" in path_lower:
            return "embedded"
        elif "radar" in path_lower:
            return "radar"
        elif "rf" in path_lower or "radio" in path_lower:
            return "rf"
        elif "ew" in path_lower or "electronic_warfare" in path_lower:
            return "ew"
        elif "ate" in path_lower or "test" in path_lower:
            return "ate"
        else:
            return "general"
    
    def _determine_security_from_path(self, file_path: str) -> str:
        """Determine security classification from file path"""
        path_lower = file_path.lower()
        
        if any(keyword in path_lower for keyword in ["classified", "secret"]):
            return "classified"
        elif any(keyword in path_lower for keyword in ["confidential", "restricted"]):
            return "confidential"
        elif any(keyword in path_lower for keyword in ["internal", "proprietary"]):
            return "internal"
        else:
            return "public"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            cursor = self.db_connection.cursor()
            
            # Document stats
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            # Domain distribution
            cursor.execute("""
                SELECT domain, COUNT(*) 
                FROM documents 
                GROUP BY domain 
                ORDER BY COUNT(*) DESC
            """)
            domain_distribution = dict(cursor.fetchall())
            
            # Security level distribution
            cursor.execute("""
                SELECT security_classification, COUNT(*) 
                FROM documents 
                GROUP BY security_classification
            """)
            security_distribution = dict(cursor.fetchall())
            
            # Recent queries
            cursor.execute("""
                SELECT COUNT(*) 
                FROM query_log 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            queries_24h = cursor.fetchone()[0]
            
            return {
                "documents": {
                    "total": total_docs,
                    "chunks": total_chunks,
                    "domain_distribution": domain_distribution,
                    "security_distribution": security_distribution
                },
                "usage": {
                    "queries_24h": queries_24h,
                    "cache_hit_rate": self._calculate_cache_hit_rate()
                },
                "system": {
                    "collections": len(self.qdrant_client.get_collections().collections),
                    "embedding_models": list(self.embedding_models.keys())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            info = self.redis_client.info("stats")
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            
            if hits + misses == 0:
                return 0.0
            
            return hits / (hits + misses)
            
        except Exception as e:
            self.logger.warning(f"Error calculating cache hit rate: {e}")
            return 0.0
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "services": {},
            "models": {},
            "data": {}
        }
        
        try:
            # Check Qdrant
            try:
                collections = self.qdrant_client.get_collections()
                health_status["services"]["qdrant"] = {
                    "status": "healthy",
                    "collections": len(collections.collections)
                }
            except Exception as e:
                health_status["services"]["qdrant"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check PostgreSQL
            try:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                health_status["services"]["postgresql"] = {"status": "healthy"}
            except Exception as e:
                health_status["services"]["postgresql"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check Redis
            try:
                self.redis_client.ping()
                health_status["services"]["redis"] = {"status": "healthy"}
            except Exception as e:
                health_status["services"]["redis"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check embedding models
            for model_name, model in self.embedding_models.items():
                try:
                    # Test embedding generation
                    test_embedding = model.encode("test", normalize_embeddings=True)
                    health_status["models"][model_name] = {
                        "status": "healthy",
                        "dimension": len(test_embedding)
                    }
                except Exception as e:
                    health_status["models"][model_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"
            
            # Check data integrity
            try:
                stats = self.get_stats()
                health_status["data"] = {
                    "documents": stats.get("documents", {}).get("total", 0),
                    "chunks": stats.get("documents", {}).get("chunks", 0),
                    "queries_24h": stats.get("usage", {}).get("queries_24h", 0)
                }
            except Exception as e:
                health_status["data"] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    async def cleanup_old_data(self, days_old: int = 30):
        """Clean up old cached data and logs"""
        try:
            # Clean old cache entries
            cache_pattern = "rag_cache:*"
            cache_keys = self.redis_client.keys(cache_pattern)
            
            # Remove expired cache entries
            for key in cache_keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No expiration set
                    self.redis_client.expire(key, 3600)  # Set 1 hour expiration
            
            # Clean old query logs
            cursor = self.db_connection.cursor()
            cursor.execute("""
                DELETE FROM query_log 
                WHERE timestamp < NOW() - INTERVAL '%s days'
            """, (days_old,))
            
            deleted_logs = cursor.rowcount
            self.db_connection.commit()
            
            self.logger.info(f"Cleaned up {deleted_logs} old query logs")
            
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
    
    def close(self):
        """Close all connections"""
        try:
            if self.db_connection:
                self.db_connection.close()
            if self.redis_client:
                self.redis_client.close()
            self.logger.info("RAG Pipeline connections closed")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()