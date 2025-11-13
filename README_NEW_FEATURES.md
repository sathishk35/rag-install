# RAG System - New Features Quick Start Guide

This guide helps you quickly start using the new features added to the RAG system.

## 🚀 Quick Start

### 1. Query Optimizer

The query optimizer automatically enhances your queries for better retrieval.

```python
from core.query_optimizer import QueryOptimizer

# Initialize
optimizer = QueryOptimizer(config)

# Optimize a query
optimized = await optimizer.optimize("How to fix i2c driver timeout bug?")

print(f"Query Type: {optimized.query_type}")  # 'troubleshooting'
print(f"Domain Hints: {optimized.domain_hints}")  # ['drivers', 'embedded']
print(f"Expanded Terms: {optimized.expanded_terms}")
```

### 2. Hybrid Search (BM25 + Dense)

Combines keyword matching with semantic search for better results.

```python
from core.hybrid_retriever import HybridRetriever

# Initialize
hybrid = HybridRetriever(
    vector_client=qdrant_client,
    embedding_model=embedding_model,
    alpha=0.7  # 70% dense, 30% sparse
)

# Search
results = await hybrid.search(
    query="kernel driver initialization",
    collection_name="code_chunks",
    top_k=10
)
```

### 3. Reranking

Rerank search results for better relevance.

```python
from core.reranker import Reranker

# Initialize
reranker = Reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda:3"
)

# Rerank results
reranked = reranker.rerank(
    query="driver initialization",
    documents=search_results,
    top_k=5
)
```

### 4. Semantic Caching

Speed up repeated queries with intelligent caching.

```python
from core.semantic_cache import SemanticCache

# Initialize
cache = SemanticCache(
    redis_client=redis_client,
    embedding_model=embedding_model,
    similarity_threshold=0.95
)

# Check cache before processing
cached = await cache.get("How to initialize driver?")

if not cached:
    result = await process_query("How to initialize driver?")
    await cache.set("How to initialize driver?", result)
```

### 5. Prompt Security

Protect against prompt injection attacks.

```python
from core.prompt_security import PromptSecurityManager

# Initialize
security = PromptSecurityManager(config)

# Validate user input
is_allowed, processed_query, analysis = security.process_query(
    query=user_input,
    user_clearance="internal"
)

if not is_allowed:
    return "Query blocked due to security concerns"

# Validate output before sending to user
sanitized_output, violations = security.validate_output(
    output=llm_response,
    user_clearance="internal"
)
```

### 6. Bugzilla Integration

Automatically sync closed bugs into the knowledge base.

```python
from integrations.bugzilla_integration import BugzillaIntegration

# Initialize
bugzilla = BugzillaIntegration(
    bugzilla_url="https://bugzilla.example.com",
    api_key=os.environ.get('BUGZILLA_API_KEY')
)

# One-time sync
stats = await bugzilla.sync_closed_bugs(
    rag_pipeline=rag_pipeline,
    since_date=datetime.now() - timedelta(days=30),
    products=['BSP', 'Drivers']
)

print(f"Synced {stats['ingested']} bugs")

# Setup automatic periodic sync (daily)
await bugzilla.setup_periodic_sync(
    rag_pipeline=rag_pipeline,
    interval_hours=24
)
```

### 7. HyDE Retrieval

Use hypothetical document embeddings for better retrieval.

```python
from core.advanced_rag import HyDERetriever

# Initialize
hyde = HyDERetriever(llm_client, embedding_model, vector_client)

# Retrieve using HyDE
results = await hyde.retrieve(
    query="How to fix i2c timeout errors?",
    collection_name="documents",
    top_k=10
)
```

### 8. Query Decomposition

Break complex queries into simpler sub-queries.

```python
from core.advanced_rag import QueryDecomposer

# Initialize
decomposer = QueryDecomposer(llm_client)

# Decompose complex query
sub_queries = await decomposer.decompose(
    "What is the driver architecture and how to implement and debug new drivers?"
)

# Process each sub-query
all_results = []
for sub_q in sub_queries:
    results = await rag_pipeline.query(sub_q.sub_query)
    all_results.extend(results)
```

### 9. Multi-Team Domain Routing

Route queries to appropriate team domains.

```python
from core.advanced_rag import MultiTeamDomainRouter

# Initialize
router = MultiTeamDomainRouter()

# Route query
routed_domains = router.route_query(
    query="How to implement FPGA-based signal processing?",
    user_context={'allowed_teams': ['hardware', 'specialized']}
)

print(f"Team: {routed_domains[0].team}")  # 'specialized'
print(f"Domain: {routed_domains[0].domain}")  # 'radar'

# Get filters for search
filters = router.get_domain_filters(routed_domains)
results = await vector_search(query, filters=filters)
```

### 10. Database Connection Pooling

Efficient database operations with connection pooling.

```python
from core.database_manager import DatabaseManager

# Initialize
db_manager = DatabaseManager(config)

# Use context manager for safe queries
with db_manager.get_cursor() as cursor:
    cursor.execute("SELECT * FROM documents WHERE domain = %s", ('drivers',))
    results = cursor.fetchall()

# Or use helper methods
results = db_manager.execute_query_dict(
    "SELECT * FROM users WHERE security_level >= %s",
    ('confidential',)
)

# Get pool statistics
stats = db_manager.get_pool_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Avg query time: {stats['avg_query_time']}")
```

## 🔧 Configuration

Add these sections to your `config/rag_config.yaml`:

```yaml
# Enable hybrid search
retrieval:
  hybrid_search:
    enabled: true
    alpha: 0.7

# Enable reranking
reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  device: "cuda:3"

# Enable semantic caching
caching:
  semantic_cache:
    enabled: true
    similarity_threshold: 0.95

# Enable security features
security:
  prompt_injection_protection:
    enabled: true
    max_threat_level: "suspicious"
  dlp:
    enabled: true

# Configure Bugzilla integration
integrations:
  bugzilla:
    enabled: true
    url: "https://bugzilla.example.com"
    sync_interval_hours: 24
```

## 📊 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_query_optimizer.py -v

# Run with coverage
pytest --cov=core --cov-report=html tests/
```

## 📈 Performance Tips

1. **Use Semantic Caching**: Enable for 10x speedup on repeated queries
2. **Tune Hybrid Search Alpha**: Adjust based on your use case (0.7 is a good default)
3. **Enable Reranking**: Significantly improves top-k relevance
4. **Use Connection Pooling**: Already enabled by default with DatabaseManager
5. **Configure Cache TTL**: Adjust based on how often your data changes

## 🔒 Security Best Practices

1. Always enable prompt injection protection
2. Enable DLP for sensitive environments
3. Set appropriate threat level thresholds
4. Monitor security violation logs
5. Regularly review audit logs

## 🐛 Troubleshooting

### Import Errors
```bash
pip install -r requirements_latest.txt --force-reinstall
```

### Redis Connection Issues
```bash
redis-cli ping  # Should return PONG
```

### Query Optimizer Not Found
The new modules are in `/core/`. Ensure your Python path includes this directory.

### Slow Performance
- Enable semantic caching
- Check database connection pool settings
- Monitor GPU utilization for reranking

## 📚 Documentation

- **Full Improvements Guide**: `docs/RAG-System-Improvements-2025.md`
- **Architecture**: `docs/RAG-System-Architecture.md`
- **Knowledge Base**: `docs/RAG-System-Knowledge-Base.md`

## 🤝 Support

For issues or questions:
1. Check the documentation in `/docs/`
2. Review logs in `/var/log/rag/`
3. Run health check: `curl http://localhost:8000/api/health`
4. Contact the RAG System Development Team

## 📝 What's New Summary

- ✅ **Query Optimizer**: Intelligent query enhancement
- ✅ **Hybrid Search**: BM25 + Dense vectors
- ✅ **Reranking**: Cross-encoder for better relevance
- ✅ **Semantic Caching**: 10x faster repeated queries
- ✅ **Prompt Security**: Injection & DLP protection
- ✅ **Bugzilla Integration**: Auto-sync bug knowledge
- ✅ **HyDE**: Hypothetical document embeddings
- ✅ **Query Decomposition**: Handle complex queries
- ✅ **Domain Routing**: Multi-team support
- ✅ **Connection Pooling**: Better scalability
- ✅ **Test Suite**: Comprehensive unit tests

Enjoy the improved RAG system! 🚀
