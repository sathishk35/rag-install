"""
Semantic Caching for RAG System
Caches query results based on semantic similarity to reduce redundant processing
"""

import logging
import hashlib
import json
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis

@dataclass
class CachedResult:
    """Container for cached query result"""
    query: str
    query_embedding: List[float]
    result: Any
    timestamp: float
    hit_count: int = 0
    metadata: Dict[str, Any] = None

class SemanticCache:
    """
    Semantic cache that stores query results and retrieves them based on similarity
    More intelligent than exact match caching
    """

    def __init__(self,
                 redis_client: redis.Redis,
                 embedding_model,
                 similarity_threshold: float = 0.95,
                 ttl: int = 3600,
                 max_cache_size: int = 10000):
        """
        Args:
            redis_client: Redis client instance
            embedding_model: Model to encode queries
            similarity_threshold: Minimum similarity to consider a cache hit
            ttl: Time to live for cache entries (seconds)
            max_cache_size: Maximum number of entries to cache
        """
        self.redis_client = redis_client
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.max_cache_size = max_cache_size
        self.logger = logging.getLogger(__name__)

        # Cache keys
        self.cache_prefix = "semantic_cache:"
        self.embedding_key = "semantic_cache:embeddings"
        self.metadata_key = "semantic_cache:metadata"

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0,
            'cache_size': 0,
            'avg_similarity_on_hit': 0.0
        }

    def _get_cache_key(self, query_hash: str) -> str:
        """Generate cache key for query"""
        return f"{self.cache_prefix}{query_hash}"

    def _hash_query(self, query: str) -> str:
        """Generate hash for query"""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def get(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Get cached result for semantically similar query

        Args:
            query: Query string
            filters: Optional filters that must match

        Returns:
            Cached result if similar query found, None otherwise
        """
        self.stats['total_queries'] += 1
        start_time = time.time()

        try:
            # Encode query
            query_embedding = self.embedding_model.encode(query).tolist()

            # Check for exact match first (faster)
            query_hash = self._hash_query(query)
            exact_key = self._get_cache_key(query_hash)

            exact_match = self.redis_client.get(exact_key)
            if exact_match:
                cached_data = json.loads(exact_match)
                self._update_hit_stats(cached_data, 1.0)
                self.stats['hits'] += 1
                self.logger.debug(f"Exact cache hit for query: {query[:50]}...")
                return cached_data['result']

            # No exact match, search for semantically similar queries
            similar_result = await self._find_similar_query(
                query_embedding,
                filters
            )

            if similar_result:
                self.stats['hits'] += 1
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Semantic cache hit (similarity={similar_result[1]:.3f}) "
                    f"in {elapsed:.3f}s"
                )
                return similar_result[0]

            # Cache miss
            self.stats['misses'] += 1
            return None

        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None

    async def _find_similar_query(self,
                                 query_embedding: List[float],
                                 filters: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Any, float]]:
        """
        Find semantically similar cached query

        Returns:
            Tuple of (cached_result, similarity_score) if found, None otherwise
        """
        try:
            # Get all cached embeddings
            cached_embeddings_raw = self.redis_client.hgetall(self.embedding_key)

            if not cached_embeddings_raw:
                return None

            # Find most similar
            best_similarity = 0.0
            best_query_hash = None

            for query_hash_bytes, embedding_json in cached_embeddings_raw.items():
                query_hash = query_hash_bytes.decode() if isinstance(query_hash_bytes, bytes) else query_hash_bytes

                try:
                    cached_embedding = json.loads(embedding_json)
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_query_hash = query_hash

                except json.JSONDecodeError:
                    continue

            # Check if similarity exceeds threshold
            if best_similarity >= self.similarity_threshold and best_query_hash:
                cache_key = self._get_cache_key(best_query_hash)
                cached_data_raw = self.redis_client.get(cache_key)

                if cached_data_raw:
                    cached_data = json.loads(cached_data_raw)

                    # Check filters match if provided
                    if filters and cached_data.get('filters') != filters:
                        return None

                    # Update hit statistics
                    self._update_hit_stats(cached_data, best_similarity)

                    return (cached_data['result'], best_similarity)

            return None

        except Exception as e:
            self.logger.error(f"Error finding similar query: {e}")
            return None

    async def set(self,
                 query: str,
                 result: Any,
                 filters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Cache query result

        Args:
            query: Query string
            result: Result to cache
            filters: Optional filters used in query
            metadata: Optional metadata
        """
        try:
            # Check cache size limit
            current_size = self.redis_client.hlen(self.embedding_key)
            if current_size >= self.max_cache_size:
                self._evict_old_entries()

            # Encode query
            query_embedding = self.embedding_model.encode(query).tolist()

            # Create cache entry
            query_hash = self._hash_query(query)
            cache_key = self._get_cache_key(query_hash)

            cached_data = {
                'query': query,
                'result': result,
                'filters': filters,
                'metadata': metadata or {},
                'timestamp': time.time(),
                'hit_count': 0
            }

            # Store in Redis
            self.redis_client.setex(
                cache_key,
                self.ttl,
                json.dumps(cached_data)
            )

            # Store embedding separately for similarity search
            self.redis_client.hset(
                self.embedding_key,
                query_hash,
                json.dumps(query_embedding)
            )

            self.stats['cache_size'] = current_size + 1
            self.logger.debug(f"Cached query: {query[:50]}...")

        except Exception as e:
            self.logger.error(f"Cache set error: {e}")

    def _update_hit_stats(self, cached_data: Dict[str, Any], similarity: float):
        """Update statistics for cache hit"""
        # Update hit count
        cached_data['hit_count'] = cached_data.get('hit_count', 0) + 1

        # Update average similarity
        total_hits = self.stats['hits'] + 1
        current_avg = self.stats['avg_similarity_on_hit']
        self.stats['avg_similarity_on_hit'] = (
            (current_avg * (total_hits - 1) + similarity) / total_hits
        )

    def _evict_old_entries(self, num_to_evict: int = 100):
        """Evict old cache entries using LRU-like strategy"""
        try:
            # Get all entries with timestamps
            all_keys = self.redis_client.keys(f"{self.cache_prefix}*")

            entries = []
            for key in all_keys:
                data_raw = self.redis_client.get(key)
                if data_raw:
                    data = json.loads(data_raw)
                    entries.append((key, data.get('timestamp', 0), data.get('hit_count', 0)))

            # Sort by hit_count (ascending) then timestamp (ascending)
            # This evicts least frequently and least recently used
            entries.sort(key=lambda x: (x[2], x[1]))

            # Evict oldest/least used entries
            for key, _, _ in entries[:num_to_evict]:
                self.redis_client.delete(key)

                # Also remove from embeddings
                query_hash = key.decode().replace(self.cache_prefix, '') if isinstance(key, bytes) else key.replace(self.cache_prefix, '')
                self.redis_client.hdel(self.embedding_key, query_hash)

            self.logger.info(f"Evicted {num_to_evict} cache entries")

        except Exception as e:
            self.logger.error(f"Error evicting entries: {e}")

    def invalidate(self, query: str):
        """Invalidate cache for specific query"""
        try:
            query_hash = self._hash_query(query)
            cache_key = self._get_cache_key(query_hash)

            self.redis_client.delete(cache_key)
            self.redis_client.hdel(self.embedding_key, query_hash)

            self.logger.debug(f"Invalidated cache for query: {query[:50]}...")

        except Exception as e:
            self.logger.error(f"Cache invalidation error: {e}")

    def clear(self):
        """Clear all cache entries"""
        try:
            # Delete all cache keys
            keys = self.redis_client.keys(f"{self.cache_prefix}*")
            if keys:
                self.redis_client.delete(*keys)

            # Clear embeddings
            self.redis_client.delete(self.embedding_key)

            # Clear metadata
            self.redis_client.delete(self.metadata_key)

            # Reset stats
            self.stats['cache_size'] = 0

            self.logger.info("Cache cleared")

        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0

        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'total_queries': self.stats['total_queries'],
            'hit_rate': round(hit_rate, 2),
            'cache_size': self.redis_client.hlen(self.embedding_key),
            'similarity_threshold': self.similarity_threshold,
            'avg_similarity_on_hit': round(self.stats['avg_similarity_on_hit'], 3),
            'ttl_seconds': self.ttl
        }

    def update_threshold(self, new_threshold: float):
        """Update similarity threshold"""
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        self.similarity_threshold = new_threshold
        self.logger.info(f"Similarity threshold updated to {new_threshold}")


class MultiLevelCache:
    """
    Multi-level caching system with different strategies for different data types
    """

    def __init__(self,
                 redis_client: redis.Redis,
                 embedding_model):
        self.redis_client = redis_client
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)

        # Different caches for different purposes
        self.query_cache = SemanticCache(
            redis_client,
            embedding_model,
            similarity_threshold=0.95,
            ttl=3600
        )

        self.embedding_cache = SemanticCache(
            redis_client,
            embedding_model,
            similarity_threshold=0.99,
            ttl=7200
        )

    async def get_query_result(self, query: str, filters: Optional[Dict] = None) -> Optional[Any]:
        """Get cached query result"""
        return await self.query_cache.get(query, filters)

    async def set_query_result(self, query: str, result: Any, filters: Optional[Dict] = None):
        """Cache query result"""
        await self.query_cache.set(query, result, filters)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        key = f"embedding:{text_hash}"

        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set_embedding(self, text: str, embedding: List[float], ttl: int = 7200):
        """Cache embedding"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        key = f"embedding:{text_hash}"

        self.redis_client.setex(key, ttl, json.dumps(embedding))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        return {
            'query_cache': self.query_cache.get_stats(),
            'embedding_cache_size': len(self.redis_client.keys("embedding:*"))
        }
