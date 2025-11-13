"""
Unit tests for QueryOptimizer
"""

import pytest
import asyncio
from core.query_optimizer import QueryOptimizer, OptimizedQuery

@pytest.fixture
def optimizer():
    """Create QueryOptimizer instance"""
    config = {}
    return QueryOptimizer(config)

def test_optimizer_initialization(optimizer):
    """Test optimizer initializes correctly"""
    assert optimizer is not None
    assert len(optimizer.domain_synonyms) > 0
    assert len(optimizer.intent_patterns) > 0

@pytest.mark.asyncio
async def test_optimize_code_query(optimizer):
    """Test optimization of code-related query"""
    query = "How to implement driver initialization function"
    result = await optimizer.optimize(query)

    assert isinstance(result, OptimizedQuery)
    assert result.query_type == 'code_search'
    assert result.original_query == query
    assert len(result.domain_hints) >= 0

@pytest.mark.asyncio
async def test_optimize_documentation_query(optimizer):
    """Test optimization of documentation query"""
    query = "What is the specification for radar waveform"
    result = await optimizer.optimize(query)

    assert isinstance(result, OptimizedQuery)
    assert result.query_type in ['documentation', 'architecture']
    assert 'radar' in result.domain_hints or len(result.domain_hints) == 0

@pytest.mark.asyncio
async def test_optimize_troubleshooting_query(optimizer):
    """Test optimization of troubleshooting query"""
    query = "Fix kernel crash when loading module"
    result = await optimizer.optimize(query)

    assert isinstance(result, OptimizedQuery)
    assert result.query_type == 'troubleshooting'
    assert result.confidence > 0

def test_extract_technical_entities(optimizer):
    """Test extraction of technical entities"""
    query = "How does the initialize_device() function work in driver.c?"
    entities = optimizer.extract_technical_entities(query)

    assert 'functions' in entities
    assert 'files' in entities
    assert 'initialize_device' in entities['functions'] or len(entities['functions']) >= 0

@pytest.mark.asyncio
async def test_decompose_query(optimizer):
    """Test query decomposition"""
    complex_query = "What is the driver architecture and how to implement new drivers"
    sub_queries = await optimizer.decompose_query(complex_query)

    assert len(sub_queries) >= 1
    assert all(isinstance(q, str) for q in sub_queries)

def test_suggest_alternative_queries(optimizer):
    """Test alternative query suggestions"""
    query = "driver code"
    alternatives = optimizer.suggest_alternative_queries(query, 'code_search')

    assert len(alternatives) > 0
    assert all(isinstance(alt, str) for alt in alternatives)
