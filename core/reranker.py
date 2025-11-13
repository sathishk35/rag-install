"""
Reranking Module for RAG System
Reranks retrieved documents using cross-encoder models for improved relevance
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import time

@dataclass
class RerankResult:
    """Container for reranked result"""
    doc_id: str
    content: str
    original_score: float
    rerank_score: float
    final_score: float
    rank: int
    metadata: Dict[str, Any]

class Reranker:
    """
    Cross-encoder based reranking model
    Provides more accurate relevance scoring by jointly encoding query and document
    """

    def __init__(self,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "cuda:3",
                 batch_size: int = 32):
        """
        Args:
            model_name: HuggingFace cross-encoder model name
            device: Device to run model on
            batch_size: Batch size for reranking
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

        # Load model
        self.model = None
        self._load_model()

        # Statistics
        self.stats = {
            'total_reranks': 0,
            'total_documents': 0,
            'total_time': 0.0
        }

    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, device=self.device)
            self.logger.info(f"Reranker model loaded: {self.model_name} on {self.device}")

        except ImportError:
            self.logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to load reranker model: {e}")
            raise

    def rerank(self,
               query: str,
               documents: List[Dict[str, Any]],
               top_k: Optional[int] = None,
               return_documents: bool = True) -> List[RerankResult]:
        """
        Rerank documents based on relevance to query

        Args:
            query: Search query
            documents: List of documents with 'id', 'content', 'score'
            top_k: Number of top results to return (None = all)
            return_documents: Whether to return full documents or just scores

        Returns:
            List of RerankResult objects sorted by rerank score
        """
        start_time = time.time()

        if not documents:
            return []

        try:
            # Prepare query-document pairs
            pairs = [[query, doc['content']] for doc in documents]

            # Get rerank scores in batches
            rerank_scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(batch)
                rerank_scores.extend(batch_scores)

            # Combine with original scores
            results = []
            for doc, rerank_score in zip(documents, rerank_scores):
                original_score = doc.get('score', 0.0)

                # Combine scores (weighted average: 0.7 rerank + 0.3 original)
                final_score = 0.7 * float(rerank_score) + 0.3 * original_score

                results.append(RerankResult(
                    doc_id=doc.get('id', ''),
                    content=doc.get('content', ''),
                    original_score=original_score,
                    rerank_score=float(rerank_score),
                    final_score=final_score,
                    rank=0,  # Will be set after sorting
                    metadata=doc.get('metadata', {})
                ))

            # Sort by final score
            results.sort(key=lambda x: x.final_score, reverse=True)

            # Update ranks
            for rank, result in enumerate(results, 1):
                result.rank = rank

            # Apply top_k if specified
            if top_k is not None:
                results = results[:top_k]

            # Update statistics
            elapsed = time.time() - start_time
            self.stats['total_reranks'] += 1
            self.stats['total_documents'] += len(documents)
            self.stats['total_time'] += elapsed

            self.logger.info(
                f"Reranked {len(documents)} documents in {elapsed:.3f}s, "
                f"returning top {len(results)}"
            )

            return results

        except Exception as e:
            self.logger.error(f"Reranking error: {e}")
            # Return original results with unchanged scores
            return [
                RerankResult(
                    doc_id=doc.get('id', ''),
                    content=doc.get('content', ''),
                    original_score=doc.get('score', 0.0),
                    rerank_score=doc.get('score', 0.0),
                    final_score=doc.get('score', 0.0),
                    rank=rank,
                    metadata=doc.get('metadata', {})
                )
                for rank, doc in enumerate(documents, 1)
            ]

    async def rerank_async(self,
                          query: str,
                          documents: List[Dict[str, Any]],
                          top_k: Optional[int] = None) -> List[RerankResult]:
        """
        Async version of rerank

        Args:
            query: Search query
            documents: List of documents
            top_k: Number of top results

        Returns:
            List of RerankResult objects
        """
        # Run reranking in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.rerank,
            query,
            documents,
            top_k
        )

    def rerank_batch(self,
                     queries: List[str],
                     documents_list: List[List[Dict[str, Any]]],
                     top_k: Optional[int] = None) -> List[List[RerankResult]]:
        """
        Rerank multiple query-document sets

        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            top_k: Number of top results per query

        Returns:
            List of RerankResult lists (one per query)
        """
        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get reranking statistics"""
        avg_time = (
            self.stats['total_time'] / self.stats['total_reranks']
            if self.stats['total_reranks'] > 0 else 0
        )
        avg_docs = (
            self.stats['total_documents'] / self.stats['total_reranks']
            if self.stats['total_reranks'] > 0 else 0
        )

        return {
            'model': self.model_name,
            'device': self.device,
            'total_reranks': self.stats['total_reranks'],
            'total_documents': self.stats['total_documents'],
            'avg_time_per_rerank': round(avg_time, 3),
            'avg_docs_per_rerank': round(avg_docs, 1)
        }


class EnsembleReranker:
    """
    Ensemble reranker combining multiple reranking strategies
    """

    def __init__(self, rerankers: List[Reranker], weights: Optional[List[float]] = None):
        """
        Args:
            rerankers: List of Reranker instances
            weights: Optional weights for each reranker (must sum to 1)
        """
        self.rerankers = rerankers
        self.logger = logging.getLogger(__name__)

        # Set weights
        if weights is None:
            self.weights = [1.0 / len(rerankers)] * len(rerankers)
        else:
            if len(weights) != len(rerankers):
                raise ValueError("Number of weights must match number of rerankers")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights

        self.logger.info(f"Ensemble reranker initialized with {len(rerankers)} models")

    def rerank(self,
               query: str,
               documents: List[Dict[str, Any]],
               top_k: Optional[int] = None) -> List[RerankResult]:
        """
        Rerank using ensemble of models

        Args:
            query: Search query
            documents: List of documents
            top_k: Number of top results

        Returns:
            List of RerankResult objects with ensemble scores
        """
        if not documents:
            return []

        # Get rerank scores from each model
        all_results = []
        for reranker in self.rerankers:
            results = reranker.rerank(query, documents, top_k=None)
            all_results.append(results)

        # Combine scores using weighted average
        doc_scores = {}
        for results, weight in zip(all_results, self.weights):
            for result in results:
                doc_id = result.doc_id
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'content': result.content,
                        'original_score': result.original_score,
                        'ensemble_score': 0.0,
                        'metadata': result.metadata
                    }
                doc_scores[doc_id]['ensemble_score'] += weight * result.rerank_score

        # Create final results
        final_results = []
        for doc_id, data in doc_scores.items():
            final_results.append(RerankResult(
                doc_id=doc_id,
                content=data['content'],
                original_score=data['original_score'],
                rerank_score=data['ensemble_score'],
                final_score=data['ensemble_score'],
                rank=0,
                metadata=data['metadata']
            ))

        # Sort by ensemble score
        final_results.sort(key=lambda x: x.final_score, reverse=True)

        # Update ranks
        for rank, result in enumerate(final_results, 1):
            result.rank = rank

        # Apply top_k
        if top_k is not None:
            final_results = final_results[:top_k]

        return final_results


class ContextualReranker:
    """
    Contextual reranker that considers document context and relationships
    """

    def __init__(self, base_reranker: Reranker):
        self.base_reranker = base_reranker
        self.logger = logging.getLogger(__name__)

    def rerank_with_context(self,
                           query: str,
                           documents: List[Dict[str, Any]],
                           context: Optional[Dict[str, Any]] = None,
                           top_k: Optional[int] = None) -> List[RerankResult]:
        """
        Rerank considering document context (e.g., same file, same function)

        Args:
            query: Search query
            documents: List of documents
            context: Optional context information
            top_k: Number of top results

        Returns:
            List of RerankResult objects with context-aware scores
        """
        # First, do base reranking
        base_results = self.base_reranker.rerank(query, documents, top_k=None)

        # Apply context-based boosting
        if context:
            for result in base_results:
                boost = 0.0

                # Boost documents from same file
                if context.get('source_file') == result.metadata.get('source_file'):
                    boost += 0.1

                # Boost documents from same domain
                if context.get('domain') == result.metadata.get('domain'):
                    boost += 0.05

                # Boost documents with same security level
                if context.get('security_level') == result.metadata.get('security_level'):
                    boost += 0.05

                # Apply boost
                result.final_score = result.rerank_score * (1 + boost)

        # Re-sort by final score
        base_results.sort(key=lambda x: x.final_score, reverse=True)

        # Update ranks
        for rank, result in enumerate(base_results, 1):
            result.rank = rank

        # Apply top_k
        if top_k is not None:
            base_results = base_results[:top_k]

        return base_results
