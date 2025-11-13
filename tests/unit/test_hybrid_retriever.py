"""
Unit tests for Hybrid Retriever
"""

import pytest
from core.hybrid_retriever import BM25Retriever, SearchResult

@pytest.fixture
def bm25():
    """Create BM25Retriever instance"""
    return BM25Retriever()

def test_bm25_initialization(bm25):
    """Test BM25 initializes correctly"""
    assert bm25 is not None
    assert bm25.k1 == 1.5
    assert bm25.b == 0.75
    assert bm25.doc_count == 0

def test_bm25_add_documents(bm25):
    """Test adding documents to BM25 index"""
    documents = [
        {'id': 'doc1', 'content': 'kernel driver initialization code'},
        {'id': 'doc2', 'content': 'device driver interface implementation'},
        {'id': 'doc3', 'content': 'radar signal processing algorithm'}
    ]

    bm25.add_documents(documents)

    assert bm25.doc_count == 3
    assert len(bm25.documents) == 3
    assert 'doc1' in bm25.documents

def test_bm25_search(bm25):
    """Test BM25 search functionality"""
    documents = [
        {'id': 'doc1', 'content': 'kernel driver initialization code for embedded systems'},
        {'id': 'doc2', 'content': 'device driver interface implementation in C'},
        {'id': 'doc3', 'content': 'radar signal processing algorithm using DSP'}
    ]

    bm25.add_documents(documents)

    # Search for driver-related content
    results = bm25.search('driver initialization', top_k=2)

    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].score > 0
    assert results[0].source == 'sparse'

def test_bm25_empty_search(bm25):
    """Test BM25 search with no documents"""
    results = bm25.search('test query')

    assert len(results) == 0

def test_bm25_get_stats(bm25):
    """Test BM25 statistics"""
    documents = [
        {'id': 'doc1', 'content': 'test document one'},
        {'id': 'doc2', 'content': 'test document two'}
    ]

    bm25.add_documents(documents)
    stats = bm25.get_stats()

    assert stats['doc_count'] == 2
    assert stats['vocabulary_size'] > 0
    assert 'avg_doc_length' in stats
