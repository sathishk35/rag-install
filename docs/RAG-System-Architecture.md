# RAG System Architecture for DP

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Client Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  OpenWebUI (Extended) │ Custom RAG UI │ API Clients │ IDE Plugins│
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                   RAG Orchestration Layer                      │
├─────────────────────────────────────────────────────────────────┤
│           LangChain + Custom RAG Pipeline                      │
│  • Query Processing    • Context Retrieval  • Response Fusion  │
│  • Security Filtering • Access Control     • Audit Logging    │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Model Layer                                  │
├─────────────────────────────────────────────────────────────────┤
│    DeepSeek-R1-70B    │    Embedding Models    │  Reranking     │
│    (via Ollama)       │   (SentenceTransformers) │   Models      │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                  Knowledge Base Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  Qdrant Vector DB │ PostgreSQL Metadata │ Redis Cache │ MinIO   │
│  (Embeddings)     │ (Structured Data)   │ (Sessions)  │ (Files) │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                        │
├─────────────────────────────────────────────────────────────────┤
│ Document Parsers │ Code Analyzers │ Chunking Engine │ Embedders │
│ • Unstructured  │ • Tree-sitter  │ • Semantic     │ • BGE-M3   │
│ • PyMuPDF       │ • AST Parser   │ • Recursive    │ • E5-large │
│ • python-docx   │ • Pygments     │ • Sliding Window│           │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Data Sources                               │
├─────────────────────────────────────────────────────────────────┤
│ Driver Source │ App Source │ Documents │ Defect Logs │ Dev Logs │
│ C/C++ Code   │ Web Code   │ PDF/Word  │ Analysis    │ Git Logs │
│ Python/MATLAB│ Databases  │ ODT Files │ Reports     │ Issues   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Vector Database: Qdrant
**Why Qdrant:**
- Excellent performance with large datasets
- Built-in support for hybrid search (dense + sparse)
- Strong security features for air-gapped environments
- Efficient memory usage with quantization
- Real-time updates and CRUD operations
- Built-in clustering for high availability

### 2. Embedding Strategy: Multi-Model Approach
**Primary:** BGE-M3 (Multi-lingual, Multi-granularity)
**Secondary:** E5-large-v2 (Code-optimized)
**Specialized:** CodeBERT for code-specific embeddings

### 3. Document Processing Pipeline
```
Input → Security Scanner → Parser → Chunker → Embedder → Store
  ↓         ↓              ↓        ↓         ↓        ↓
Files → Classification → Extraction → Segments → Vectors → Qdrant
```

### 4. Security & Access Control
- Document classification (Public/Internal/Confidential/Classified)
- Role-based access control (RBAC)
- Audit logging for all queries
- Data sanitization for sensitive content
- Encrypted storage and transmission

### 5. Real-time Updates
- File system watchers for new documents
- Incremental embedding updates
- Cache invalidation strategies
- Version control integration

## Hardware Allocation

### GPU Distribution (4x L40 24GB):
- **GPU 0-1:** DeepSeek-R1-70B (48GB total)
- **GPU 2:** Embedding models (24GB)
- **GPU 3:** Reranking + Backup (24GB)

### Memory Usage (128GB RAM):
- **32GB:** Qdrant vector database
- **16GB:** PostgreSQL metadata
- **8GB:** Redis cache
- **16GB:** Document processing workers
- **32GB:** System + buffers
- **24GB:** Reserved for peak loads

### Storage Strategy:
- **SSD (2TB):** Hot data (recent documents, frequent queries)
- **HDD (10TB+):** Cold storage (archived documents, backups)
- **NVMe (500GB):** Database indices and cache
