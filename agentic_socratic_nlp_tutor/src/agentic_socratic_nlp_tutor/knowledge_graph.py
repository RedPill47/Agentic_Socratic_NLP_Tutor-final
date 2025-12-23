"""
Enhanced Knowledge Graph

Structured knowledge graph with Concept dataclass and metadata.
Supports prerequisite relationships, learning paths, and inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import deque


@dataclass
class Concept:
    """Represents an NLP concept with metadata."""
    name: str
    difficulty: str = "intermediate"  # "beginner", "intermediate", "advanced"
    topic_area: str = "General NLP"
    description: str = ""
    learning_time_minutes: int = 60
    keywords: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Concept):
            return self.name == other.name
        return False


@dataclass
class Relationship:
    """Represents a prerequisite relationship between concepts."""
    prerequisite: str  # Name of prerequisite concept
    concept: str       # Name of concept that requires it
    strength: float = 1.0  # 0-1, how critical is this prerequisite
    
    def __hash__(self):
        return hash((self.prerequisite, self.concept))
    
    def __eq__(self, other):
        if isinstance(other, Relationship):
            return self.prerequisite == other.prerequisite and self.concept == other.concept
        return False


class EnhancedPrerequisiteGraph:
    """
    Enhanced prerequisite graph using Concept objects.
    
    Features:
    - Structured concepts with metadata
    - Prerequisite relationships with strength weights
    - Learning path generation
    - Mastery inference
    - Topic detection
    """
    
    def __init__(self):
        """Initialize empty graph."""
        # Concept storage: name -> Concept
        self.concepts: Dict[str, Concept] = {}
        
        # Prerequisite relationships: concept -> Set[prerequisite]
        self.prerequisites: Dict[str, Set[str]] = {}
        
        # Reverse: prerequisite -> Set[concepts that require it]
        self.dependents: Dict[str, Set[str]] = {}
        
        # Relationship strengths: (prerequisite, concept) -> strength
        self.relationship_strengths: Dict[Tuple[str, str], float] = {}
        
        # Initialize with default NLP concepts
        self._initialize_default_concepts()

    @property
    def graph(self) -> Dict[str, List[str]]:
        """
        Compatibility layer to mirror SimplePrerequisiteGraph API.
        Returns a mapping of concept -> list of prerequisites.
        """
        return {name: list(prs) for name, prs in self.prerequisites.items()}
    
    def _initialize_default_concepts(self):
        """Initialize graph with default NLP concepts."""
        # Foundation concepts
        # Add Text Preprocessing as a parent concept
        self.add_concept(Concept(
            name="Text Preprocessing",
            difficulty="beginner",
            topic_area="Text Preprocessing",
            description="Preparing raw text for NLP tasks: tokenization, normalization, cleaning",
            learning_time_minutes=60,
            keywords=["text", "processing", "preprocessing", "normalization", "cleaning", "preprocess"]
        ))
        
        self.add_concept(Concept(
            name="Tokenization",
            difficulty="beginner",
            topic_area="Text Preprocessing",
            description="Breaking text into tokens (words, subwords, characters)",
            learning_time_minutes=30,
            keywords=["token", "tokenize", "split", "word"]
        ))
        
        # Make Tokenization a prerequisite of Text Preprocessing (or vice versa - Tokenization is part of Text Preprocessing)
        # Actually, Text Preprocessing is the parent, so Tokenization doesn't require it
        # But we can add Text Preprocessing as a prerequisite for concepts that need preprocessing
        
        self.add_concept(Concept(
            name="Neural Networks",
            difficulty="beginner",
            topic_area="Deep Learning",
            description="Basic neural network architecture",
            learning_time_minutes=120,
            keywords=["neural", "network", "perceptron", "layer"]
        ))
        
        self.add_concept(Concept(
            name="Probability",
            difficulty="beginner",
            topic_area="Mathematics",
            description="Basic probability theory",
            learning_time_minutes=90,
            keywords=["probability", "distribution", "likelihood"]
        ))
        
        # Text representations
        self.add_concept(Concept(
            name="Bag of Words",
            difficulty="beginner",
            topic_area="Text Representations",
            description="Simple word frequency representation",
            learning_time_minutes=45,
            keywords=["bag", "words", "frequency", "count"]
        ))
        
        self.add_concept(Concept(
            name="Word Embeddings",
            difficulty="intermediate",
            topic_area="Text Representations",
            description="Dense vector representations of words",
            learning_time_minutes=90,
            keywords=["embedding", "vector", "word", "representation"]
        ))
        
        self.add_concept(Concept(
            name="TF-IDF",
            difficulty="intermediate",
            topic_area="Text Representations",
            description="Term frequency-inverse document frequency",
            learning_time_minutes=60,
            keywords=["tfidf", "tf-idf", "frequency", "document"]
        ))
        
        # Word embedding models
        self.add_concept(Concept(
            name="Word2Vec",
            difficulty="intermediate",
            topic_area="Word Embeddings",
            description="Word embeddings using skip-gram or CBOW",
            learning_time_minutes=120,
            keywords=["word2vec", "skip-gram", "cbow", "mikolov"]
        ))
        
        self.add_concept(Concept(
            name="GloVe",
            difficulty="intermediate",
            topic_area="Word Embeddings",
            description="Global vectors for word representation",
            learning_time_minutes=90,
            keywords=["glove", "global", "vectors"]
        ))
        
        self.add_concept(Concept(
            name="FastText",
            difficulty="intermediate",
            topic_area="Word Embeddings",
            description="Word embeddings with subword information",
            learning_time_minutes=90,
            keywords=["fasttext", "subword", "character"]
        ))
        
        # Sequence models
        self.add_concept(Concept(
            name="RNN",
            difficulty="intermediate",
            topic_area="Sequence Models",
            description="Recurrent Neural Networks for sequences",
            learning_time_minutes=120,
            keywords=["rnn", "recurrent", "sequence", "temporal"]
        ))
        
        self.add_concept(Concept(
            name="LSTM",
            difficulty="intermediate",
            topic_area="Sequence Models",
            description="Long Short-Term Memory networks",
            learning_time_minutes=150,
            keywords=["lstm", "long", "short", "memory", "gate"]
        ))
        
        self.add_concept(Concept(
            name="GRU",
            difficulty="intermediate",
            topic_area="Sequence Models",
            description="Gated Recurrent Unit",
            learning_time_minutes=120,
            keywords=["gru", "gated", "recurrent"]
        ))
        
        self.add_concept(Concept(
            name="Sequence-to-Sequence",
            difficulty="intermediate",
            topic_area="Sequence Models",
            description="Encoder-decoder architecture for sequences",
            learning_time_minutes=180,
            keywords=["seq2seq", "encoder", "decoder", "sequence"]
        ))
        
        # Attention and transformers
        self.add_concept(Concept(
            name="Attention Mechanisms",
            difficulty="advanced",
            topic_area="Attention & Transformers",
            description="Mechanism to focus on relevant parts of input",
            learning_time_minutes=150,
            keywords=["attention", "focus", "weight", "relevance"]
        ))
        
        self.add_concept(Concept(
            name="Self-Attention",
            difficulty="advanced",
            topic_area="Attention & Transformers",
            description="Attention mechanism within a single sequence",
            learning_time_minutes=120,
            keywords=["self-attention", "self", "attention"]
        ))
        
        self.add_concept(Concept(
            name="Multi-Head Attention",
            difficulty="advanced",
            topic_area="Attention & Transformers",
            description="Multiple attention heads in parallel",
            learning_time_minutes=90,
            keywords=["multi-head", "attention", "parallel"]
        ))
        
        self.add_concept(Concept(
            name="Positional Encoding",
            difficulty="advanced",
            topic_area="Attention & Transformers",
            description="Encoding position information in sequences",
            learning_time_minutes=60,
            keywords=["positional", "encoding", "position", "order"]
        ))
        
        self.add_concept(Concept(
            name="Layer Normalization",
            difficulty="advanced",
            topic_area="Attention & Transformers",
            description="Normalization technique for training stability",
            learning_time_minutes=60,
            keywords=["layer", "normalization", "norm", "batch"]
        ))
        
        self.add_concept(Concept(
            name="Transformer",
            difficulty="advanced",
            topic_area="Attention & Transformers",
            description="Transformer architecture with self-attention",
            learning_time_minutes=240,
            keywords=["transformer", "attention", "vaswani", "architecture"]
        ))
        
        # Transformer models
        self.add_concept(Concept(
            name="BERT",
            difficulty="advanced",
            topic_area="Transformer Models",
            description="Bidirectional Encoder Representations from Transformers",
            learning_time_minutes=180,
            keywords=["bert", "bidirectional", "encoder", "masked"]
        ))
        
        self.add_concept(Concept(
            name="GPT",
            difficulty="advanced",
            topic_area="Transformer Models",
            description="Generative Pre-trained Transformer",
            learning_time_minutes=180,
            keywords=["gpt", "generative", "pre-trained", "decoder"]
        ))
        
        self.add_concept(Concept(
            name="T5",
            difficulty="advanced",
            topic_area="Transformer Models",
            description="Text-to-Text Transfer Transformer",
            learning_time_minutes=150,
            keywords=["t5", "text-to-text", "transfer"]
        ))
        
        # NLP Tasks and Applications
        self.add_concept(Concept(
            name="Dependency Parsing",
            difficulty="intermediate",
            topic_area="NLP Tasks",
            description="Analyzing grammatical structure by identifying relationships between words",
            learning_time_minutes=120,
            keywords=["dependency", "parsing", "parse", "syntax", "grammar", "tree", "constituency"]
        ))
        
        self.add_concept(Concept(
            name="Named Entity Recognition",
            difficulty="intermediate",
            topic_area="NLP Tasks",
            description="Identifying and classifying named entities in text",
            learning_time_minutes=90,
            keywords=["ner", "named", "entity", "recognition", "person", "location", "organization"]
        ))
        
        self.add_concept(Concept(
            name="Part-of-Speech Tagging",
            difficulty="intermediate",
            topic_area="NLP Tasks",
            description="Labeling words with their grammatical parts of speech",
            learning_time_minutes=60,
            keywords=["pos", "tagging", "part", "speech", "noun", "verb", "adjective"]
        ))
        
        self.add_concept(Concept(
            name="Sentiment Analysis",
            difficulty="intermediate",
            topic_area="NLP Tasks",
            description="Determining emotional tone or sentiment in text",
            learning_time_minutes=90,
            keywords=["sentiment", "analysis", "emotion", "polarity", "positive", "negative"]
        ))
        
        # Auto-generated concepts from RAG extraction (68 concepts)
        # Generated by: scripts/extract_topics_from_rag.py
        # Date: December 14, 2024
        # Note: Some concepts may already exist with slight name variations
        self._load_extracted_concepts()
        
        # Add prerequisite relationships
        self._add_default_relationships()
    
    def _load_extracted_concepts(self):
        """Load extracted concepts from generated code file."""
        import os
        import logging
        from pathlib import Path
        
        logger = logging.getLogger(__name__)
        
        # Get path to generated code file
        current_file = Path(__file__).resolve()
        # From: agentic_socratic_nlp_tutor/src/agentic_socratic_nlp_tutor/knowledge_graph.py
        # To project root: go up 4 levels
        project_root = current_file.parent.parent.parent.parent
        code_file = project_root / "data" / "generated_graph_code.py"
        
        if not code_file.exists():
            logger.debug(f"📚 [KnowledgeGraph] Generated code file not found at {code_file}, skipping extracted concepts")
            return
        
        try:
            logger.info(f"📚 [KnowledgeGraph] Loading extracted concepts from {code_file}")
            
            # Read and execute the generated code
            with open(code_file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Remove the comment lines at the top
            lines = code.split('\n')
            # Find where actual code starts (after comments)
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('self.add_concept'):
                    start_idx = i
                    break
            
            if start_idx == 0:
                logger.warning(f"📚 [KnowledgeGraph] No code found in {code_file}")
                return
            
            # Count concepts before
            concepts_before = len(self.concepts)
            
            # Find the base indentation level (first non-empty line)
            base_indent = None
            for line in lines[start_idx:]:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    if base_indent is None or indent < base_indent:
                        base_indent = indent
                    break
            
            if base_indent is None:
                logger.warning(f"📚 [KnowledgeGraph] No code found in {code_file}")
                return
            
            # Strip base indentation from all lines (preserve relative indentation)
            code_lines = []
            for line in lines[start_idx:]:
                if line.strip():  # Non-empty line
                    # Remove base indentation, keep relative indentation
                    if len(line) > base_indent:
                        code_lines.append(line[base_indent:])
                    else:
                        code_lines.append(line.lstrip())
                else:
                    code_lines.append('')  # Preserve empty lines for structure
            
            # Separate concepts and prerequisites by parsing blocks
            concept_blocks = []
            prerequisite_lines = []
            
            i = 0
            while i < len(code_lines):
                line = code_lines[i]
                if line.strip().startswith('self.add_concept'):
                    # Collect multi-line Concept() call
                    block = [line]
                    i += 1
                    paren_count = line.count('(') - line.count(')')
                    while i < len(code_lines) and paren_count > 0:
                        block.append(code_lines[i])
                        paren_count += code_lines[i].count('(') - code_lines[i].count(')')
                        i += 1
                    concept_blocks.append('\n'.join(block))
                elif line.strip().startswith('self.add_prerequisite'):
                    prerequisite_lines.append(line)
                    i += 1
                else:
                    i += 1
            
            # Pass 1: Add all concepts first
            exec_globals = {'self': self, 'Concept': Concept}
            if concept_blocks:
                try:
                    for block in concept_blocks:
                        exec(block, exec_globals)
                    logger.debug(f"📚 [KnowledgeGraph] Added {len(concept_blocks)} concepts")
                except Exception as e:
                    logger.warning(f"⚠️ [KnowledgeGraph] Error adding some concepts: {e}")
            
            # Pass 2: Add prerequisites (skip ones that fail)
            prerequisites_added = 0
            prerequisites_skipped = 0
            for line in prerequisite_lines:
                try:
                    exec(line, exec_globals)
                    prerequisites_added += 1
                except ValueError as e:
                    # Prerequisite doesn't exist - skip it
                    prerequisites_skipped += 1
                    logger.debug(f"⚠️ [KnowledgeGraph] Skipping prerequisite: {e}")
                except Exception as e:
                    logger.warning(f"⚠️ [KnowledgeGraph] Error adding prerequisite: {e}")
            
            # Count concepts after
            concepts_after = len(self.concepts)
            added_count = concepts_after - concepts_before
            
            logger.info(f"📚 [KnowledgeGraph] Loaded {added_count} extracted concepts (total: {concepts_after})")
            if prerequisites_added > 0:
                logger.info(f"📚 [KnowledgeGraph] Added {prerequisites_added} prerequisite relationships")
            if prerequisites_skipped > 0:
                logger.debug(f"📚 [KnowledgeGraph] Skipped {prerequisites_skipped} prerequisites (concepts not in graph)")
            
        except Exception as e:
            # Log the error instead of silently failing
            logger.error(f"❌ [KnowledgeGraph] Failed to load extracted concepts: {e}", exc_info=True)
            # Continue without extracted concepts
    
    def _add_default_relationships(self):
        """Add default prerequisite relationships."""
        # Foundation relationships
        self.add_prerequisite("Bag of Words", "Tokenization")
        self.add_prerequisite("Word Embeddings", "Neural Networks")
        self.add_prerequisite("Word Embeddings", "Tokenization")
        self.add_prerequisite("TF-IDF", "Tokenization")
        
        # Word embedding models
        self.add_prerequisite("Word2Vec", "Word Embeddings")
        self.add_prerequisite("Word2Vec", "Neural Networks")
        self.add_prerequisite("GloVe", "Word Embeddings")
        self.add_prerequisite("FastText", "Word Embeddings")
        
        # Sequence models
        self.add_prerequisite("RNN", "Neural Networks")
        self.add_prerequisite("LSTM", "RNN")
        self.add_prerequisite("GRU", "RNN")
        self.add_prerequisite("Sequence-to-Sequence", "RNN")
        self.add_prerequisite("Sequence-to-Sequence", "LSTM")
        
        # Attention and transformers
        self.add_prerequisite("Attention Mechanisms", "Sequence-to-Sequence")
        self.add_prerequisite("Self-Attention", "Attention Mechanisms")
        self.add_prerequisite("Multi-Head Attention", "Self-Attention")
        self.add_prerequisite("Transformer", "Self-Attention")
        self.add_prerequisite("Transformer", "Multi-Head Attention")
        self.add_prerequisite("Transformer", "Positional Encoding")
        self.add_prerequisite("Transformer", "Layer Normalization")
        
        # Transformer models
        self.add_prerequisite("BERT", "Transformer")
        self.add_prerequisite("BERT", "Self-Attention")
        self.add_prerequisite("GPT", "Transformer")
        self.add_prerequisite("GPT", "Self-Attention")
        self.add_prerequisite("T5", "Transformer")
        
        # NLP Tasks
        self.add_prerequisite("Dependency Parsing", "Tokenization")
        self.add_prerequisite("Dependency Parsing", "Neural Networks")
        self.add_prerequisite("Part-of-Speech Tagging", "Tokenization")
        self.add_prerequisite("Named Entity Recognition", "Tokenization")
        self.add_prerequisite("Named Entity Recognition", "Word Embeddings")
        self.add_prerequisite("Sentiment Analysis", "Word Embeddings")
    
    def add_concept(self, concept: Concept):
        """Add a concept to the graph."""
        self.concepts[concept.name] = concept
        if concept.name not in self.prerequisites:
            self.prerequisites[concept.name] = set()
        if concept.name not in self.dependents:
            self.dependents[concept.name] = set()
    
    def add_prerequisite(self, concept: str, prerequisite: str, strength: float = 1.0):
        """
        Add a prerequisite relationship.
        
        Args:
            concept: Concept that requires the prerequisite
            prerequisite: Required concept
            strength: How critical (0-1, default 1.0)
        """
        if concept not in self.concepts:
            raise ValueError(f"Concept '{concept}' not found")
        if prerequisite not in self.concepts:
            raise ValueError(f"Prerequisite '{prerequisite}' not found")
        
        self.prerequisites[concept].add(prerequisite)
        self.dependents[prerequisite].add(concept)
        self.relationship_strengths[(prerequisite, concept)] = strength
    
    def detect_topic(self, text: str) -> Optional[str]:
        """
        Detect topic from text using keyword matching.
        
        Args:
            text: Input text
            
        Returns:
            Concept name if detected, None otherwise
        """
        text_lower = text.lower()
        
        # First pass: Check for exact phrase matches (highest priority)
        # This handles cases like "Sequence Labeling" vs "Sequence-to-Sequence"
        exact_matches = []
        for concept_name in self.concepts.keys():
            concept_lower = concept_name.lower()
            # Check if the full concept name appears as a phrase
            if concept_lower in text_lower:
                # Prefer longer/more specific matches
                exact_matches.append((concept_name, len(concept_name)))
        
        if exact_matches:
            # Return the longest match (most specific)
            exact_matches.sort(key=lambda x: x[1], reverse=True)
            return exact_matches[0][0]
        
        # Second pass: Check multi-word concepts where all words appear
        # This is more specific than single-word matches
        multi_word_matches = []
        for concept_name in self.concepts.keys():
            if " " in concept_name or "-" in concept_name:
                words = concept_name.lower().replace("-", " ").split()
                if len(words) > 1 and all(word in text_lower for word in words):
                    multi_word_matches.append((concept_name, len(words)))
        
        if multi_word_matches:
            # Return the concept with most words (most specific)
            multi_word_matches.sort(key=lambda x: x[1], reverse=True)
            return multi_word_matches[0][0]
        
        # Third pass: Check keywords (weighted scoring)
        best_match = None
        best_score = 0
        
        for concept in self.concepts.values():
            # Count keyword matches
            score = sum(1 for keyword in concept.keywords if keyword.lower() in text_lower)
            # Boost score if concept name words appear (even if not exact match)
            concept_words = concept.name.lower().replace("-", " ").split()
            name_matches = sum(1 for word in concept_words if len(word) > 3 and word in text_lower)
            total_score = score + (name_matches * 0.5)  # Partial credit for name word matches
            
            if total_score > best_score:
                best_score = total_score
                best_match = concept.name
        
        return best_match if best_score > 0 else None
    
    def get_prerequisites(self, concept: str) -> List[str]:
        """
        Get direct prerequisites for a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            List of prerequisite names
        """
        return list(self.prerequisites.get(concept, set()))
    
    def get_all_prerequisites(self, concept: str) -> List[str]:
        """
        Get all prerequisites (transitive closure).
        
        Args:
            concept: Concept name
            
        Returns:
            Topologically sorted list of all prerequisites
        """
        if concept not in self.concepts:
            return []
        
        visited = set()
        result = []
        
        def dfs(concept_name: str):
            if concept_name in visited:
                return
            visited.add(concept_name)
            
            for prereq in self.prerequisites.get(concept_name, set()):
                dfs(prereq)
            
            if concept_name != concept:  # Don't include the concept itself
                result.append(concept_name)
        
        dfs(concept)
        
        # Reverse to get topological order (foundations first)
        return result[::-1]
    
    def get_gaps(self, concept: str, mastered: List[str]) -> List[str]:
        """
        Find prerequisite gaps.
        
        Args:
            concept: Target concept
            mastered: List of mastered concept names
            
        Returns:
            List of missing prerequisites
        """
        all_prereqs = set(self.get_all_prerequisites(concept))
        mastered_set = set(mastered)
        gaps = all_prereqs - mastered_set
        return list(gaps)
    
    def get_middle_prerequisite(self, concept: str) -> Optional[str]:
        """
        Get a prerequisite in the middle of the prerequisite chain.
        Useful for binary search in onboarding.
        
        Args:
            concept: Target concept
            
        Returns:
            Middle prerequisite name, or None if no prerequisites
        """
        all_prereqs = self.get_all_prerequisites(concept)
        if not all_prereqs:
            return None
        
        # Return middle element
        mid_idx = len(all_prereqs) // 2
        return all_prereqs[mid_idx]
    
    def infer_mastery(self, demonstrated_concept: str) -> List[str]:
        """
        Infer mastery of prerequisites from demonstrated knowledge.
        
        Args:
            demonstrated_concept: Concept the student demonstrated knowledge of
            
        Returns:
            List of concepts (demonstrated + all prerequisites)
        """
        if demonstrated_concept not in self.concepts:
            return []
        
        all_prereqs = self.get_all_prerequisites(demonstrated_concept)
        return [demonstrated_concept] + all_prereqs
    
    def get_learning_path(self, start: str, end: str) -> List[str]:
        """
        Get optimal learning path from start to end concept.
        
        Args:
            start: Starting concept (should be prerequisite of end)
            end: Target concept
            
        Returns:
            List of concepts in learning order
        """
        if start not in self.concepts or end not in self.concepts:
            return []
        
        # Get all prerequisites of end
        all_prereqs = set(self.get_all_prerequisites(end))
        
        if start not in all_prereqs and start != end:
            # Start is not a prerequisite, return path from start's prerequisites to end
            start_prereqs = set(self.get_all_prerequisites(start))
            path_concepts = (all_prereqs - start_prereqs) | {start, end}
        else:
            path_concepts = all_prereqs | {start, end}
        
        # Topological sort
        visited = set()
        result = []
        
        def visit(concept_name: str):
            if concept_name in visited or concept_name not in path_concepts:
                return
            visited.add(concept_name)
            
            for prereq in self.prerequisites.get(concept_name, set()):
                if prereq in path_concepts:
                    visit(prereq)
            
            result.append(concept_name)
        
        visit(end)
        return result
    
    def get_concept(self, name: str) -> Optional[Concept]:
        """Get concept by name."""
        return self.concepts.get(name)
    
    def get_all_concepts(self) -> List[Concept]:
        """Get all concepts."""
        return list(self.concepts.values())
    
    def get_concepts_by_difficulty(self, difficulty: str) -> List[Concept]:
        """Get all concepts of a specific difficulty."""
        return [c for c in self.concepts.values() if c.difficulty == difficulty]
    
    def get_concepts_by_topic(self, topic_area: str) -> List[Concept]:
        """Get all concepts in a topic area."""
        return [c for c in self.concepts.values() if c.topic_area == topic_area]
