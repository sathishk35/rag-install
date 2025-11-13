"""
Hybrid Retrieval System
Combines dense vector search (semantic) with sparse BM25 search (keyword matching)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from collections import defaultdict
import re
import math

@dataclass
class SearchResult:
    """Container for search results"""
    doc_id: str
    content: str
    score: float
    source: str  # 'dense', 'sparse', or 'hybrid'
    metadata: Dict[str, Any]
    rank: int = 0

class BM25Retriever:
    """
    BM25 (Best Matching 25) sparse retrieval implementation
    Good for keyword and exact term matching
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0.75 is standard)
        """
        self.k1 = k1
        self.b = b
        self.logger = logging.getLogger(__name__)

        # Document statistics
        self.doc_count = 0
        self.avg_doc_length = 0
        self.doc_lengths = {}
        self.doc_freqs = {}  # Document frequency for each term
        self.idf_cache = {}  # IDF scores cache

        # Inverted index: term -> [(doc_id, term_freq), ...]
        self.inverted_index = defaultdict(list)

        # Document store: doc_id -> content
        self.documents = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms"""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _calculate_idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency for a term"""
        if term in self.idf_cache:
            return self.idf_cache[term]

        df = self.doc_freqs.get(term, 0)
        if df == 0:
            idf = 0
        else:
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

        self.idf_cache[term] = idf
        return idf

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the BM25 index

        Args:
            documents: List of dicts with 'id', 'content', and optional 'metadata'
        """
        for doc in documents:
            doc_id = doc['id']
            content = doc['content']

            # Store document
            self.documents[doc_id] = content

            # Tokenize
            tokens = self._tokenize(content)
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length

            # Count term frequencies
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            # Update inverted index and document frequencies
            for term, freq in term_freqs.items():
                self.inverted_index[term].append((doc_id, freq))
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Update statistics
        self.doc_count = len(self.documents)
        if self.doc_count > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.doc_count

        # Clear IDF cache when documents are added
        self.idf_cache.clear()

        self.logger.info(f"Added {len(documents)} documents to BM25 index. Total: {self.doc_count}")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for documents matching the query

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects sorted by score
        """
        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Calculate BM25 scores for all documents
        scores = defaultdict(float)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            idf = self._calculate_idf(term)

            # For each document containing this term
            for doc_id, term_freq in self.inverted_index[term]:
                doc_length = self.doc_lengths[doc_id]

                # BM25 formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )

                scores[doc_id] += idf * (numerator / denominator)

        # Sort by score and get top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Create SearchResult objects
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            results.append(SearchResult(
                doc_id=doc_id,
                content=self.documents[doc_id],
                score=score,
                source='sparse',
                metadata={'bm25_score': score},
                rank=rank
            ))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get BM25 index statistics"""
        return {
            'doc_count': self.doc_count,
            'avg_doc_length': self.avg_doc_length,
            'vocabulary_size': len(self.inverted_index),
            'total_terms': sum(len(postings) for postings in self.inverted_index.values())
        }


class HybridRetriever:
    """
    Hybrid retrieval system combining dense (vector) and sparse (BM25) search
    """

    def __init__(self,
                 vector_client,
                 embedding_model,
                 alpha: float = 0.7,
                 k1: float = 1.5,
                 b: float = 0.75):
        """
        Args:
            vector_client: Qdrant client for dense search
            embedding_model: Embedding model for query encoding
            alpha: Weight for dense search (0-1). sparse weight = 1-alpha
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.vector_client = vector_client
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)

        # Initialize BM25 retriever
        self.bm25 = BM25Retriever(k1=k1, b=b)

        # Cache for loaded documents
        self.indexed_docs = {}

    async def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for hybrid search

        Args:
            documents: List of documents with 'id', 'content', 'vector', 'metadata'
        """
        # Add to BM25 index
        bm25_docs = [
            {'id': doc['id'], 'content': doc['content']}
            for doc in documents
        ]
        self.bm25.add_documents(bm25_docs)

        # Store document metadata
        for doc in documents:
            self.indexed_docs[doc['id']] = doc

        self.logger.info(f"Indexed {len(documents)} documents for hybrid search")

    async def search(self,
                    query: str,
                    collection_name: str,
                    top_k: int = 10,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Perform hybrid search combining dense and sparse retrieval

        Args:
            query: Search query
            collection_name: Qdrant collection name
            top_k: Number of results to return
            filters: Optional filters for vector search

        Returns:
            List of SearchResult objects sorted by combined score
        """
        # Perform dense (vector) search
        dense_results = await self._dense_search(query, collection_name, top_k * 2, filters)

        # Perform sparse (BM25) search
        sparse_results = self.bm25.search(query, top_k * 2)

        # Combine results
        combined_results = self._combine_results(dense_results, sparse_results, top_k)

        return combined_results

    async def _dense_search(self,
                           query: str,
                           collection_name: str,
                           top_k: int,
                           filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform dense vector search"""
        try:
            # Encode query
            query_vector = self.embedding_model.encode(query).tolist()

            # Search Qdrant
            search_params = {
                'collection_name': collection_name,
                'query_vector': query_vector,
                'limit': top_k
            }

            if filters:
                search_params['query_filter'] = filters

            search_results = self.vector_client.search(**search_params)

            # Convert to SearchResult objects
            results = []
            for rank, hit in enumerate(search_results, 1):
                results.append(SearchResult(
                    doc_id=str(hit.id),
                    content=hit.payload.get('content', ''),
                    score=hit.score,
                    source='dense',
                    metadata={
                        'vector_score': hit.score,
                        **hit.payload
                    },
                    rank=rank
                ))

            return results

        except Exception as e:
            self.logger.error(f"Dense search error: {e}")
            return []

    def _combine_results(self,
                        dense_results: List[SearchResult],
                        sparse_results: List[SearchResult],
                        top_k: int) -> List[SearchResult]:
        """
        Combine dense and sparse results using weighted score fusion

        Uses Reciprocal Rank Fusion (RRF) combined with score normalization
        """
        # Normalize scores for both result sets
        dense_normalized = self._normalize_scores(dense_results)
        sparse_normalized = self._normalize_scores(sparse_results)

        # Combine scores
        combined_scores = {}

        # Add dense scores (weighted by alpha)
        for result in dense_normalized:
            doc_id = result.doc_id
            combined_scores[doc_id] = {
                'score': self.alpha * result.score,
                'content': result.content,
                'metadata': result.metadata,
                'dense_score': result.score,
                'sparse_score': 0.0
            }

        # Add sparse scores (weighted by 1-alpha)
        for result in sparse_normalized:
            doc_id = result.doc_id
            if doc_id in combined_scores:
                combined_scores[doc_id]['score'] += (1 - self.alpha) * result.score
                combined_scores[doc_id]['sparse_score'] = result.score
            else:
                combined_scores[doc_id] = {
                    'score': (1 - self.alpha) * result.score,
                    'content': result.content,
                    'metadata': result.metadata,
                    'dense_score': 0.0,
                    'sparse_score': result.score
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:top_k]

        # Create final SearchResult objects
        final_results = []
        for rank, (doc_id, data) in enumerate(sorted_results, 1):
            final_results.append(SearchResult(
                doc_id=doc_id,
                content=data['content'],
                score=data['score'],
                source='hybrid',
                metadata={
                    **data['metadata'],
                    'hybrid_score': data['score'],
                    'dense_score': data['dense_score'],
                    'sparse_score': data['sparse_score'],
                    'alpha': self.alpha
                },
                rank=rank
            ))

        return final_results

    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores to 0-1 range using min-max normalization"""
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score == min_score:
            for result in results:
                result.score = 1.0
            return results

        # Normalize
        normalized_results = []
        for result in results:
            normalized_score = (result.score - min_score) / (max_score - min_score)
            normalized_result = SearchResult(
                doc_id=result.doc_id,
                content=result.content,
                score=normalized_score,
                source=result.source,
                metadata=result.metadata,
                rank=result.rank
            )
            normalized_results.append(normalized_result)

        return normalized_results

    def set_alpha(self, alpha: float):
        """Update the dense/sparse weight balance"""
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
        self.logger.info(f"Alpha updated to {alpha} (dense={alpha}, sparse={1-alpha})")

    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid retriever statistics"""
        return {
            'alpha': self.alpha,
            'dense_weight': self.alpha,
            'sparse_weight': 1 - self.alpha,
            'bm25_stats': self.bm25.get_stats(),
            'indexed_documents': len(self.indexed_docs)
        }
