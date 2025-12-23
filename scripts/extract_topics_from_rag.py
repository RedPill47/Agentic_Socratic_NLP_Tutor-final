"""
Extract Topics from RAG and Map to Knowledge Graph

This script:
1. Queries the vector store to extract all topics/concepts mentioned
2. Uses LLM to identify unique NLP concepts
3. Adds missing concepts to the knowledge graph
4. Optionally infers prerequisite relationships

Usage:
    python scripts/extract_topics_from_rag.py
    # Or use backend wrapper: python backend/scripts/sync_knowledge_graph.py
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Set, Dict, Optional
from collections import Counter
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Set UTF-8 encoding for Windows compatibility (handle emojis)
# Only wrap if not already wrapped and not being captured by subprocess
if sys.platform == "win32":
    try:
        import io
        # Check if stdout is a TextIOWrapper (already wrapped) or if it's being captured
        if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
            try:
                # Test if we can write to it
                sys.stdout.write('')
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            except (ValueError, OSError):
                # File is closed or being captured, don't wrap
                pass
        if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, io.TextIOWrapper):
            try:
                # Test if we can write to it
                sys.stderr.write('')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            except (ValueError, OSError):
                # File is closed or being captured, don't wrap
                pass
    except Exception:
        pass  # If encoding setup fails, continue anyway

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "agentic_socratic_nlp_tutor" / "src"))

# Load environment variables early (before imports that might need them)
# Try multiple .env locations
env_paths = [
    project_root / ".env",
    project_root.parent / ".env",
    Path.cwd() / ".env",
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        env_loaded = True
        break

# If no .env found, try default load_dotenv() behavior
if not env_loaded:
    load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from agentic_socratic_nlp_tutor.knowledge_graph import EnhancedPrerequisiteGraph, Concept

# Paths
DB_PATH = project_root / "data" / "chroma_db"
OUTPUT_FILE = project_root / "data" / "extracted_topics.json"
CODE_OUTPUT_FILE = project_root / "data" / "generated_graph_code.py"


class TopicExtractor:
    """Extract NLP topics from vector store and map to knowledge graph."""
    
    def __init__(self):
        """Initialize topic extractor."""
        self.llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Load vector store
        if not DB_PATH.exists():
            raise FileNotFoundError(f"Vector store not found at {DB_PATH}")
        
        embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = Chroma(
            persist_directory=str(DB_PATH),
            embedding_function=embedding_function
        )
        
        # Load knowledge graph
        self.knowledge_graph = EnhancedPrerequisiteGraph()
        self.existing_concepts = set(self.knowledge_graph.concepts.keys())
        
        print(f"[OK] Loaded knowledge graph with {len(self.existing_concepts)} concepts")
        print(f"[OK] Loaded vector store from {DB_PATH}")
    
    def get_all_chunks(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all chunks from vector store.
        
        Args:
            limit: Maximum number of chunks to retrieve (None = all)
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        try:
            # Get collection count
            collection = self.vector_store._collection
            total_count = collection.count()
            
            print(f"📊 Total chunks in vector store: {total_count}")
            
            if limit:
                total_count = min(total_count, limit)
            
            # Sample chunks (can't get all at once efficiently)
            # Use diverse queries to get representative sample
            sample_queries = [
                "tokenization",
                "neural networks",
                "transformers",
                "attention",
                "embeddings",
                "classification",
                "parsing",
                "language modeling",
                "sequence models",
                "text representation"
            ]
            
            all_chunks = []
            seen_ids = set()
            
            # Get chunks from diverse queries
            for query in sample_queries:
                try:
                    results = self.vector_store.similarity_search(query, k=20)
                    for doc in results:
                        doc_id = id(doc.page_content)  # Simple deduplication
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            all_chunks.append({
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "source": doc.metadata.get("source_file", "unknown")
                            })
                except Exception as e:
                    print(f"⚠️ Error querying '{query}': {e}")
            
            # If we need more, do random sampling
            if len(all_chunks) < total_count and not limit:
                # Get more chunks using additional queries
                additional_queries = [
                    "NLP", "natural language", "machine learning",
                    "deep learning", "text processing", "syntax", "semantics"
                ]
                for query in additional_queries:
                    if len(all_chunks) >= total_count:
                        break
                    try:
                        results = self.vector_store.similarity_search(query, k=30)
                        for doc in results:
                            doc_id = id(doc.page_content)
                            if doc_id not in seen_ids:
                                seen_ids.add(doc_id)
                                all_chunks.append({
                                    "content": doc.page_content,
                                    "metadata": doc.metadata,
                                    "source": doc.metadata.get("source_file", "unknown")
                                })
                    except Exception as e:
                        print(f"⚠️ Error querying '{query}': {e}")
            
            print(f"[OK] Retrieved {len(all_chunks)} unique chunks")
            return all_chunks[:total_count] if limit else all_chunks
            
        except Exception as e:
            print(f"[ERROR] Error getting chunks: {e}")
            return []
    
    async def extract_topics_from_chunks(self, chunks: List[Dict], batch_size: int = 5) -> Dict[str, Dict]:
        """
        Extract NLP topics from chunks using structured LLM analysis.
        Similar to PDF ingestion tool's metadata generation.
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Number of chunks to process per LLM call
            
        Returns:
            Dictionary mapping topic names to their metadata
        """
        all_topics = {}  # topic_name -> {metadata}
        
        print(f"\n[INFO] Extracting topics from {len(chunks)} chunks using structured LLM analysis...")
        print("   (This uses LLM to build accurate knowledge graph)")
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Build batch text with metadata if available
            batch_text = []
            for j, chunk in enumerate(batch):
                chunk_info = f"Chunk {j+1}:\n"
                if chunk.get('metadata', {}).get('concept'):
                    chunk_info += f"Concept: {chunk['metadata']['concept']}\n"
                if chunk.get('metadata', {}).get('topic'):
                    chunk_info += f"Topic: {chunk['metadata']['topic']}\n"
                chunk_info += f"Content: {chunk['content'][:800]}..."
                batch_text.append(chunk_info)
            
            batch_text_str = "\n\n---\n\n".join(batch_text)
            
            try:
                prompt = f"""Analyze these educational text chunks and extract NLP concepts with structured information.

TEXT CHUNKS:
{batch_text_str}

For EACH unique NLP concept mentioned, provide:
CONCEPT: [concept name - use standard NLP terminology, e.g., "Dependency Parsing", "Named Entity Recognition"]
DIFFICULTY: [beginner/intermediate/advanced]
TOPIC_AREA: [e.g., "NLP Tasks", "Text Representations", "Sequence Models", "Attention & Transformers"]
DESCRIPTION: [one sentence description]
KEYWORDS: [comma-separated, 3-5 keywords]
PREREQUISITES: [comma-separated prerequisite concepts, or "none" if foundational]

Format each concept as:
---
CONCEPT: [name]
DIFFICULTY: [level]
TOPIC_AREA: [area]
DESCRIPTION: [description]
KEYWORDS: [keywords]
PREREQUISITES: [prerequisites]
---

IMPORTANT:
- Extract ALL significant NLP concepts mentioned
- Use standard, canonical names (e.g., "Dependency Parsing" not "dependency parse")
- Be specific about prerequisites based on what the text implies
- If a concept appears multiple times, provide the most complete information
- Only include concepts that are clearly explained or defined in the chunks

Return concepts:"""
                
                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse structured response
                concepts = self._parse_concept_extraction(response_text)
                
                # Merge into all_topics (keep most complete info)
                for concept_name, concept_data in concepts.items():
                    if concept_name not in all_topics:
                        all_topics[concept_name] = concept_data
                    else:
                        # Merge: keep more complete description, combine keywords
                        existing = all_topics[concept_name]
                        if len(concept_data.get('description', '')) > len(existing.get('description', '')):
                            existing['description'] = concept_data['description']
                        existing_keywords = set(existing.get('keywords', []))
                        new_keywords = set(concept_data.get('keywords', []))
                        existing['keywords'] = list(existing_keywords | new_keywords)
                
                if (i // batch_size + 1) % 5 == 0:
                    print(f"   Processed {i + len(batch)}/{len(chunks)} chunks, found {len(all_topics)} unique concepts...")
                    
            except Exception as e:
                print(f"⚠️ Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"[OK] Extracted {len(all_topics)} unique concepts with metadata")
        return all_topics
    
    def _parse_concept_extraction(self, response_text: str) -> Dict[str, Dict]:
        """
        Parse structured concept extraction response.
        
        Args:
            response_text: LLM response with structured concept information
            
        Returns:
            Dictionary mapping concept names to metadata
        """
        concepts = {}
        current_concept = None
        current_data = {}
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line == '---':
                # Save current concept if complete
                if current_concept and current_data:
                    concepts[current_concept] = current_data
                current_concept = None
                current_data = {}
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                if key == 'CONCEPT':
                    # Save previous concept if exists
                    if current_concept and current_data:
                        concepts[current_concept] = current_data
                    current_concept = value
                    current_data = {}
                elif key == 'DIFFICULTY':
                    current_data['difficulty'] = value.lower()
                elif key == 'TOPIC_AREA':
                    current_data['topic_area'] = value
                elif key == 'DESCRIPTION':
                    current_data['description'] = value
                elif key == 'KEYWORDS':
                    # Parse comma-separated keywords
                    keywords = [k.strip() for k in value.split(',') if k.strip()]
                    current_data['keywords'] = keywords
                elif key == 'PREREQUISITES':
                    if value.lower() not in ['none', 'n/a', '']:
                        prereqs = [p.strip() for p in value.split(',') if p.strip()]
                        current_data['prerequisites'] = prereqs
                    else:
                        current_data['prerequisites'] = []
        
        # Save last concept
        if current_concept and current_data:
            concepts[current_concept] = current_data
        
        return concepts
    
    def filter_new_topics(self, topics: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Filter out topics that already exist in knowledge graph.
        
        Args:
            topics: Dictionary mapping topic names to metadata
            
        Returns:
            Dictionary of new topics not in graph
        """
        new_topics = {}
        
        for topic, metadata in topics.items():
            # Check exact match
            if topic in self.existing_concepts:
                continue
            
            # Check case-insensitive match
            topic_lower = topic.lower()
            if any(existing.lower() == topic_lower for existing in self.existing_concepts):
                continue
            
            # Check if it's a variation of existing concept
            is_variation = False
            for existing in self.existing_concepts:
                if topic_lower in existing.lower() or existing.lower() in topic_lower:
                    is_variation = True
                    break
            
            if not is_variation:
                new_topics[topic] = metadata
        
        print(f"📊 New topics to add: {len(new_topics)} (out of {len(topics)} total)")
        return new_topics
    
    async def enrich_topic_metadata(self, topic: str, existing_metadata: Optional[Dict] = None) -> Optional[Dict]:
        """
        Use LLM to enrich topic with metadata (difficulty, keywords, description).
        If existing_metadata is provided, use it as base and only fill missing fields.
        
        Args:
            topic: Topic name
            existing_metadata: Optional existing metadata from extraction
            
        Returns:
            Dictionary with metadata or None
        """
        try:
            # If we already have metadata from extraction, use it
            if existing_metadata and all(k in existing_metadata for k in ['difficulty', 'topic_area', 'description', 'keywords']):
                return existing_metadata
            
            # Otherwise, enrich with LLM
            base_info = ""
            if existing_metadata:
                base_info = f"\nExisting information:\n{json.dumps(existing_metadata, indent=2)}\n"
            
            prompt = f"""For the NLP concept "{topic}", provide complete metadata:
1. Difficulty level: "beginner", "intermediate", or "advanced"
2. Topic area: e.g., "NLP Tasks", "Text Representations", "Sequence Models"
3. Brief description (one sentence)
4. Keywords (comma-separated, 3-5 keywords)
5. Learning time estimate in minutes

{base_info}

Format as JSON:
{{
    "difficulty": "...",
    "topic_area": "...",
    "description": "...",
    "keywords": ["...", "..."],
    "learning_time_minutes": 60
}}"""
            
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            metadata = json.loads(response.choices[0].message.content)
            
            # Merge with existing if provided
            if existing_metadata:
                metadata = {**existing_metadata, **metadata}
            
            # Validate
            if metadata.get("difficulty") not in ["beginner", "intermediate", "advanced"]:
                metadata["difficulty"] = "intermediate"
            
            if "learning_time_minutes" not in metadata:
                metadata["learning_time_minutes"] = 60
            
            return metadata
            
        except Exception as e:
            print(f"⚠️ Error enriching metadata for '{topic}': {e}")
            return existing_metadata  # Return existing if enrichment fails
    
    async def infer_prerequisites(self, topic: str, topic_metadata: Dict, all_topics: Dict[str, Dict]) -> List[str]:
        """
        Infer prerequisites for a topic using LLM with context from all topics.
        
        Args:
            topic: Topic name
            topic_metadata: Metadata for this topic (may contain prerequisites from extraction)
            all_topics: Dictionary of all available topics with metadata
            
        Returns:
            List of prerequisite topic names (both from extraction and inferred)
        """
        try:
            # Start with prerequisites from extraction if available
            extracted_prereqs = topic_metadata.get('prerequisites', [])
            
            # Get all available concepts (existing + new topics)
            all_available = set(self.existing_concepts) | set(all_topics.keys())
            all_available_str = ", ".join(sorted(list(all_available))[:30])
            
            # Get topic description for context
            description = topic_metadata.get('description', '')
            
            prompt = f"""For the NLP concept "{topic}", determine its prerequisite concepts.

Concept: {topic}
Description: {description}

Available concepts (existing + newly extracted):
{all_available_str}

Extracted prerequisites (may be incomplete): {', '.join(extracted_prereqs) if extracted_prereqs else 'none'}

Analyze the concept and determine:
1. What foundational concepts must be understood first?
2. What related concepts provide necessary background?
3. Are the extracted prerequisites correct and complete?

Return ONLY a comma-separated list of prerequisite concept names.
- Use exact names from the available concepts list
- Include both extracted prerequisites (if valid) and any additional ones
- Return "none" only if truly no prerequisites are needed

Prerequisites:"""
            
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            prereqs_text = response.choices[0].message.content.strip()
            
            if prereqs_text.lower() in ["none", "n/a", "no prerequisites"]:
                return []
            
            # Parse prerequisites
            prereqs = [p.strip() for p in prereqs_text.split(",")]
            
            # Filter to only include concepts that exist (existing or in new topics)
            valid_prereqs = [p for p in prereqs if p in all_available]
            
            # Combine with extracted prerequisites (deduplicate)
            all_prereqs = list(set(extracted_prereqs + valid_prereqs))
            
            return all_prereqs
            
        except Exception as e:
            print(f"⚠️ Error inferring prerequisites for '{topic}': {e}")
            # Return extracted prerequisites as fallback
            return topic_metadata.get('prerequisites', [])
    
    async def add_topics_to_graph(self, topics: Dict[str, Dict], infer_prereqs: bool = True) -> Dict:
        """
        Add topics to knowledge graph with metadata.
        
        Args:
            topics: Dictionary mapping topic names to metadata
            infer_prereqs: Whether to infer prerequisite relationships
            
        Returns:
            Dictionary with results
        """
        results = {
            "added": [],
            "failed": [],
            "skipped": []
        }
        
        print(f"\n[INFO] Adding {len(topics)} topics to knowledge graph...")
        
        # First pass: add all concepts
        for i, (topic, topic_metadata) in enumerate(sorted(topics.items()), 1):
            print(f"\n[{i}/{len(topics)}] Processing: {topic}")
            
            try:
                # Enrich with metadata (use existing if available)
                metadata = await self.enrich_topic_metadata(topic, topic_metadata)
                if not metadata:
                    results["failed"].append(topic)
                    continue
                
                # Create concept
                concept = Concept(
                    name=topic,
                    difficulty=metadata.get("difficulty", "intermediate"),
                    topic_area=metadata.get("topic_area", "General NLP"),
                    description=metadata.get("description", ""),
                    learning_time_minutes=metadata.get("learning_time_minutes", 60),
                    keywords=metadata.get("keywords", [])
                )
                
                # Add to graph
                self.knowledge_graph.add_concept(concept)
                
                # Save to results
                results["added"].append({
                    "topic": topic,
                    "difficulty": concept.difficulty,
                    "topic_area": concept.topic_area,
                    "description": concept.description,
                    "keywords": concept.keywords,
                    "learning_time_minutes": concept.learning_time_minutes,
                    "prerequisites": []  # Will be filled in pass 2
                })
                
                print(f"   [OK] Added concept: {topic} ({concept.difficulty}, {concept.topic_area})")
                
            except Exception as e:
                print(f"   [ERROR] Failed to add '{topic}': {e}")
                results["failed"].append(topic)
        
        # Second pass: add prerequisite relationships (after all concepts are added)
        if infer_prereqs:
            print(f"\n🔗 Inferring prerequisite relationships...")
            for i, (topic, topic_metadata) in enumerate(sorted(topics.items()), 1):
                if topic in results["failed"]:
                    continue
                
                try:
                    # Infer prerequisites with full context
                    prereqs = await self.infer_prerequisites(topic, topic_metadata, topics)
                    
                    added_prereqs = []
                    for prereq in prereqs:
                        try:
                            # Check if prerequisite exists (existing or newly added)
                            if prereq in self.knowledge_graph.concepts:
                                self.knowledge_graph.add_prerequisite(topic, prereq)
                                added_prereqs.append(prereq)
                            else:
                                print(f"   ⚠️ Prerequisite '{prereq}' not found in graph, skipping")
                        except Exception as e:
                            print(f"   ⚠️ Could not add prerequisite {prereq} → {topic}: {e}")
                    
                    if added_prereqs:
                        print(f"   [OK] Added {len(added_prereqs)} prerequisites for {topic}")
                    
                    # Update results with prerequisites
                    for item in results["added"]:
                        if item["topic"] == topic:
                            item["prerequisites"] = added_prereqs
                            break
                        
                except Exception as e:
                    print(f"   ⚠️ Error inferring prerequisites for '{topic}': {e}")
        
        return results
    
    def save_results(self, results: Dict, output_file: Path):
        """Save extraction results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Saved results to {output_file}")
    
    def generate_code_for_graph(self, results: Dict) -> str:
        """
        Generate Python code to add concepts to knowledge graph.
        
        Args:
            results: Results dictionary with added concepts
            
        Returns:
            Python code string to add to knowledge_graph.py
        """
        code_lines = []
        code_lines.append("\n# Auto-generated concepts from RAG extraction")
        code_lines.append("# Add this code to _initialize_default_concepts() method")
        code_lines.append("")
        
        for item in results.get("added", []):
            topic = item["topic"]
            difficulty = item.get("difficulty", "intermediate")
            topic_area = item.get("topic_area", "General NLP")
            description = item.get("description", "")
            keywords = item.get("keywords", [])
            learning_time = item.get("learning_time_minutes", 60)
            prereqs = item.get("prerequisites", [])
            
            # Generate concept code
            code_lines.append(f"        self.add_concept(Concept(")
            code_lines.append(f'            name="{topic}",')
            code_lines.append(f'            difficulty="{difficulty}",')
            code_lines.append(f'            topic_area="{topic_area}",')
            if description:
                # Escape quotes in description
                desc_escaped = description.replace('"', '\\"')
                code_lines.append(f'            description="{desc_escaped}",')
            else:
                code_lines.append(f'            description="",  # TODO: Add description')
            code_lines.append(f'            learning_time_minutes={learning_time},')
            if keywords:
                keywords_str = ", ".join([f'"{k}"' for k in keywords])
                code_lines.append(f'            keywords=[{keywords_str}]')
            else:
                code_lines.append(f'            keywords=[]  # TODO: Add keywords')
            code_lines.append(f"        ))")
            code_lines.append("")
            
            # Generate prerequisite relationships
            for prereq in prereqs:
                code_lines.append(f'        self.add_prerequisite("{topic}", "{prereq}")')
        
        return "\n".join(code_lines)
    
    def save_code_to_file(self, code: str, output_file: Path):
        """Save generated code to file."""
        with open(output_file, 'w') as f:
            f.write(code)
        print(f"[OK] Generated code saved to {output_file}")
        print("   [INFO] Review and add to knowledge_graph.py")


async def main(auto_confirm: bool = False):
    """Main extraction pipeline.
    
    Args:
        auto_confirm: If True, automatically confirm adding topics without user input
    """
    try:
        print("=" * 60)
        print("TOPIC EXTRACTION FROM RAG TO KNOWLEDGE GRAPH")
        print("=" * 60)
        
        # Verify environment variables
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not found in environment variables")
            print("   Please ensure .env file exists in project root with OPENAI_API_KEY set")
            return
        
        extractor = TopicExtractor()
        
        # Step 1: Get chunks from vector store
        print("\n[STEP 1] Retrieving chunks from vector store...")
        chunks = extractor.get_all_chunks(limit=500)  # Limit for testing, remove for full extraction
        
        if not chunks:
            print("[ERROR] No chunks found!")
            return
        
        # Step 2: Extract topics from chunks using structured LLM analysis
        print("\n[STEP 2] Extracting topics from chunks using structured LLM analysis...")
        all_topics = await extractor.extract_topics_from_chunks(chunks, batch_size=5)
        
        # Step 3: Filter new topics
        print("\n🔎 Step 3: Filtering new topics...")
        new_topics = extractor.filter_new_topics(all_topics)
        
        if not new_topics:
            print("[OK] No new topics to add - knowledge graph is complete!")
            return
        
        print(f"\n[INFO] New topics found: {len(new_topics)}")
        for topic in sorted(new_topics.keys())[:20]:
            metadata = new_topics[topic]
            difficulty = metadata.get('difficulty', 'unknown')
            topic_area = metadata.get('topic_area', 'unknown')
            print(f"   • {topic} ({difficulty}, {topic_area})")
        if len(new_topics) > 20:
            print(f"   ... and {len(new_topics) - 20} more")
        
        # Step 4: Ask user for confirmation (unless auto_confirm)
        if not auto_confirm:
            print(f"\n[PROMPT] Add {len(new_topics)} new topics to knowledge graph? (y/n): ", end="")
            response = input().strip().lower()
            
            if response != 'y':
                print("[INFO] Cancelled by user")
                return
        else:
            print(f"\n[OK] Auto-confirming: Adding {len(new_topics)} new topics to knowledge graph...")
        
        # Step 5: Add topics to graph
        print("\n[STEP 4] Adding topics to knowledge graph...")
        results = await extractor.add_topics_to_graph(new_topics, infer_prereqs=True)
        
        # Step 6: Save results
        print("\n[STEP 5] Saving results...")
        extractor.save_results(results, OUTPUT_FILE)
        
        # Step 7: Generate code for knowledge graph
        print("\n[STEP 6] Generating code for knowledge graph...")
        code = extractor.generate_code_for_graph(results)
        extractor.save_code_to_file(code, CODE_OUTPUT_FILE)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"[OK] Added: {len(results['added'])} topics")
        print(f"[ERROR] Failed: {len(results['failed'])} topics")
        print(f"📊 Total concepts in graph: {len(extractor.knowledge_graph.concepts)}")
        print(f"[OK] Results saved to: {OUTPUT_FILE}")
        print(f"[OK] Generated code: {CODE_OUTPUT_FILE}")
        print("=" * 60)
        
        # Show added topics
        if results['added']:
            print("\n[INFO] Added Topics:")
            for item in results['added'][:10]:
                print(f"   • {item['topic']} ({item['difficulty']}, {item['topic_area']})")
            if len(results['added']) > 10:
                print(f"   ... and {len(results['added']) - 10} more")
    
    except FileNotFoundError as e:
        try:
            print(f"ERROR: {e}", file=sys.stderr)
            print("   Please ensure the vector store exists at data/chroma_db/", file=sys.stderr)
        except (ValueError, OSError):
            # Fallback if stdout/stderr are closed
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)
    except Exception as e:
        try:
            print(f"ERROR: {e}", file=sys.stderr)
            import traceback
            print("\nFull traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        except (ValueError, OSError):
            # Fallback if stdout/stderr are closed
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract topics from RAG and add to knowledge graph")
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Automatically confirm adding topics without user input (for background sync)"
    )
    
    args = parser.parse_args()
    asyncio.run(main(auto_confirm=args.auto_confirm))

