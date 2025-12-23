"""
Response-Level Semantic Caching

Caches LLM responses based on semantic similarity of queries.
Reduces LLM calls by 20-30% for similar questions.
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class CachedResponse:
    """Cached response entry."""
    query: str
    response: str
    query_embedding: List[float]
    created_at: datetime
    hit_count: int = 0
    metadata: Dict = field(default_factory=dict)


class ResponseCache:
    """
    Semantic cache for LLM responses.
    
    Uses sentence embeddings to find similar queries and return cached responses
    if similarity is above threshold.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_cache_size: int = 100,
        ttl_hours: int = 24
    ):
        """
        Initialize response cache.
        
        Args:
            similarity_threshold: Minimum cosine similarity to consider queries similar (0-1)
            max_cache_size: Maximum number of cached responses
            ttl_hours: Time-to-live for cache entries (hours)
        """
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl = timedelta(hours=ttl_hours)
        
        # Cache storage: query_hash -> CachedResponse
        self.cache: Dict[str, CachedResponse] = {}
        
        # Initialize embeddings model (same as RAG for consistency)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embeddings_available = True
            except Exception as e:
                print(f"⚠️ Failed to load embedding model: {e}")
                self.embeddings_available = False
        else:
            self.embeddings_available = False
            print("⚠️ Sentence transformers not available - semantic caching disabled")
    
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query (for exact match first check)."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _embed_query(self, query: str) -> List[float]:
        """Get embedding for query."""
        if not self.embeddings_available:
            return []
        try:
            embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"⚠️ Embedding generation failed: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now - entry.created_at > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def _evict_oldest(self):
        """Evict oldest entry if cache is full."""
        if len(self.cache) >= self.max_cache_size:
            # Remove entry with lowest hit count (or oldest if tied)
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: (self.cache[k].hit_count, self.cache[k].created_at)
            )
            del self.cache[oldest_key]
    
    def get(self, query: str) -> Optional[str]:
        """
        Get cached response if similar query exists.
        
        Args:
            query: User query
            
        Returns:
            Cached response if found, None otherwise
        """
        if not self.embeddings_available:
            return None
        
        # Cleanup expired entries
        self._cleanup_expired()
        
        # First check: exact match (fast)
        query_hash = self._get_query_hash(query)
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            entry.hit_count += 1
            return entry.response
        
        # Second check: semantic similarity
        query_embedding = self._embed_query(query)
        if not query_embedding:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for entry in self.cache.values():
            if not entry.query_embedding:
                continue
            
            similarity = self._cosine_similarity(query_embedding, entry.query_embedding)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            best_match.hit_count += 1
            return best_match.response
        
        return None
    
    def put(self, query: str, response: str, metadata: Optional[Dict] = None):
        """
        Cache a query-response pair.
        
        Args:
            query: User query
            response: LLM response
            metadata: Optional metadata (topic, difficulty, etc.)
        """
        if not self.embeddings_available:
            return
        
        # Cleanup expired entries
        self._cleanup_expired()
        
        # Evict if needed
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
        
        # Create cache entry
        query_embedding = self._embed_query(query)
        query_hash = self._get_query_hash(query)
        
        entry = CachedResponse(
            query=query,
            response=response,
            query_embedding=query_embedding,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        self.cache[query_hash] = entry
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_hits = sum(entry.hit_count for entry in self.cache.values())
        return {
            "size": len(self.cache),
            "max_size": self.max_cache_size,
            "total_hits": total_hits,
            "avg_hits_per_entry": total_hits / len(self.cache) if self.cache else 0,
            "embeddings_available": self.embeddings_available
        }
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()

