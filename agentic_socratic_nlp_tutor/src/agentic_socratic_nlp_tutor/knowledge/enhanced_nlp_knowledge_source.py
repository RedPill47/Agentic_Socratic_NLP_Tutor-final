"""
Enhanced NLP Knowledge Source for CrewAI

This custom knowledge source:
1. Connects to existing ChromaDB vector store
2. Implements advanced retrieval (query expansion, hybrid search, re-ranking)
3. Integrates seamlessly with CrewAI's knowledge system
4. Provides automatic context to agents without explicit tool calls
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from crewai import LLM
from pydantic import Field, PrivateAttr


class EnhancedNLPKnowledgeSource(BaseKnowledgeSource):
    """
    Custom knowledge source that wraps existing ChromaDB and adds advanced retrieval features.
    
    This source:
    - Uses your existing ChromaDB (no migration needed)
    - Implements query expansion, hybrid search, and re-ranking
    - Works automatically with CrewAI agents (no tool calls needed)
    - Leverages CrewAI's query rewriting for better results
    """
    
    db_path: Optional[str] = Field(
        default=None,
        description="Path to ChromaDB. If None, uses default project location."
    )
    
    # Private attributes for lazy loading
    _embeddings: Optional[Any] = PrivateAttr(default=None)
    _vectorstore: Optional[Any] = PrivateAttr(default=None)
    _llm: Optional[Any] = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default db_path if not provided
        if self.db_path is None:
            current_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))
            self.db_path = os.path.join(project_root, "data", "chroma_db")
    
    def _get_embeddings(self):
        """Lazy load embeddings."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings
    
    def _get_vectorstore(self):
        """Lazy load vector store."""
        if self._vectorstore is None:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(
                    f"Knowledge base not found at {self.db_path}. "
                    "Please run the PDF ingestion script first."
                )
            
            self._vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self._get_embeddings()
            )
        return self._vectorstore
    
    def _get_llm(self):
        """Lazy load LLM for query expansion and re-ranking."""
        if self._llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            
            self._llm = LLM(
                model="openai/gpt-5-mini",
                # Note: GPT-5-mini only supports default temperature (1), temperature parameter not supported
                api_key=api_key
            )
        return self._llm
    
    def load_content(self) -> Dict[Any, str]:
        """
        Load content from existing ChromaDB.
        
        Since the ChromaDB is already populated and embedded,
        we return an empty dict. The actual retrieval happens in query().
        """
        # We don't need to load content here because:
        # 1. ChromaDB already has embedded content
        # 2. We'll query it directly in the query() method
        # 3. This avoids re-embedding on every kickoff
        return {}
    
    def validate_content(self, content: Any) -> str:
        """
        Validate and format content for storage.
        
        Since we're using an existing ChromaDB and querying directly,
        we don't need to validate content here. This method is required
        by BaseKnowledgeSource but we return empty string as we query
        the database directly.
        
        Args:
            content: Content to validate (not used in our case)
            
        Returns:
            Validated content as string
        """
        # Since we query ChromaDB directly, we don't need to validate content
        # Return empty string as placeholder
        return ""
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Step 1: Expand query into multiple search terms using LLM.
        """
        try:
            llm = self._get_llm()
            
            prompt = f"""Expand this NLP learning query into search terms.

QUERY: "{query}"

Provide 3-5 search terms (one per line) that would help find relevant educational content.
Use standard NLP terminology. Be concise.

Example:
QUERY: "how do computers understand words"
word embeddings
semantic representation
word vectors
tokenization
distributed representations

Now expand the query above:"""
            
            response = llm.call([{"role": "user", "content": prompt}])
            
            # Parse response into list
            terms = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            # Always include original query
            if query not in terms:
                terms.insert(0, query)
            
            return terms[:5]  # Limit to 5 terms
            
        except Exception as e:
            print(f"Warning: Query expansion failed: {e}")
            return [query]
    
    def _semantic_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Step 2a: Semantic search using embeddings with MMR for diversity.
        """
        try:
            vectorstore = self._get_vectorstore()
            
            # Use MMR for diversity
            results = vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=k * 3
            )
            
            return results
            
        except Exception as e:
            print(f"Warning: Semantic search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Step 2b: Keyword search using similarity search.
        """
        try:
            vectorstore = self._get_vectorstore()
            
            results = vectorstore.similarity_search(
                query,
                k=k,
                filter=None
            )
            
            return results
            
        except Exception as e:
            print(f"Warning: Keyword search failed: {e}")
            return []
    
    def _hybrid_search(self, expanded_queries: List[str], k: int = 15) -> List[Document]:
        """
        Step 2: Perform hybrid search combining semantic and keyword results.
        """
        all_docs = []
        seen_content = set()
        
        # Search with each expanded query
        for query in expanded_queries:
            # Semantic search
            semantic_results = self._semantic_search(query, k=k//2)
            
            # Keyword search
            keyword_results = self._keyword_search(query, k=k//2)
            
            # Merge results, avoiding duplicates
            for doc in semantic_results + keyword_results:
                content_hash = doc.page_content[:100]
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        return all_docs
    
    def _rerank_with_llm(self, query: str, documents: List[Document], top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Step 3: Re-rank documents using LLM to score relevance.
        Returns list of (document, relevance_score) tuples.
        """
        try:
            llm = self._get_llm()
            
            ranked_docs = []
            
            for doc in documents[:10]:  # Only re-rank top 10 to save time
                # Create relevance prompt
                prompt = f"""Rate how relevant this text is to the student's query.

STUDENT QUERY: "{query}"

TEXT CONTENT:
{doc.page_content[:500]}...

Rate relevance from 0-10 where:
- 10 = Directly answers the query with clear explanation
- 7-9 = Highly relevant, contains useful information
- 4-6 = Somewhat relevant, tangentially related
- 1-3 = Barely relevant
- 0 = Not relevant

Respond with ONLY a single number (0-10):"""
                
                try:
                    response = llm.call([{"role": "user", "content": prompt}])
                    score = float(response.strip())
                    score = max(0, min(10, score))  # Clamp to 0-10
                    ranked_docs.append((doc, score))
                except:
                    # If scoring fails, give neutral score
                    ranked_docs.append((doc, 5.0))
            
            # Sort by score descending
            ranked_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k
            return ranked_docs[:top_k]
            
        except Exception as e:
            print(f"Warning: Re-ranking failed: {e}")
            # Return top_k without scores
            return [(doc, 5.0) for doc in documents[:top_k]]
    
    def query(self, query: str, k: int = 3) -> List[str]:
        """
        Main query method that CrewAI calls automatically.
        
        This implements the full advanced retrieval pipeline:
        1. Query expansion
        2. Hybrid search
        3. Re-ranking
        4. Return formatted results
        
        Args:
            query: The search query (may already be rewritten by CrewAI)
            k: Number of results to return
            
        Returns:
            List of document content strings
        """
        try:
            # Step 1: Expand query
            expanded_queries = self._expand_query(query)
            
            # Step 2: Hybrid search
            candidates = self._hybrid_search(expanded_queries, k=k*3)
            
            if not candidates:
                return [f"No relevant information found for '{query}' in the knowledge base."]
            
            # Step 3: Re-rank
            ranked_docs = self._rerank_with_llm(query, candidates, top_k=k)
            
            # Step 4: Format and return
            # CrewAI will format these, but we can add metadata
            results = []
            for doc, score in ranked_docs:
                # Include source information in the content
                source_info = []
                if doc.metadata.get("source_file"):
                    source_info.append(f"Source: {doc.metadata['source_file']}")
                if doc.metadata.get("page_number"):
                    source_info.append(f"Page {doc.metadata['page_number']}")
                
                citation = " | ".join(source_info) if source_info else ""
                formatted_content = f"[{citation}]\n{doc.page_content}" if citation else doc.page_content
                results.append(formatted_content)
            
            return results
            
        except Exception as e:
            return [f"Error retrieving from knowledge base: {str(e)}"]
    
    def add(self) -> None:
        """
        Add method required by BaseKnowledgeSource.
        
        Since we're using an existing ChromaDB, we don't need to add content here.
        The ChromaDB is already populated by the PDF ingestion script.
        """
        # Mark as processed without actually adding content
        # (since ChromaDB is already populated)
        self.chunks = []  # Empty chunks since we query directly
        # Don't call _save_documents() since we're using existing DB

