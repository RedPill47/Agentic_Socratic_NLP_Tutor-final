"""
Enhanced PDF Ingestion Script with AI-Powered Metadata Generation and Incremental Processing

This script:
1. Tracks processed files to avoid reprocessing
2. Only processes new PDFs
3. Incrementally adds new content to existing vector store
4. Generates rich metadata (concept, explanation, example, difficulty tags)
5. Supports downloading PDFs from URLs

Usage:
    python src/ingest_pdfs_enhanced.py
"""

import os
import sys
import glob
import json
import hashlib
import requests
from pathlib import Path
from typing import List, Set
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Add the agentic_socratic_nlp_tutor package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agentic_socratic_nlp_tutor', 'src'))

from agentic_socratic_nlp_tutor.tools.metadata_generator import (
    batch_generate_metadata,
    generate_topic_summary
)

# Load environment variables
load_dotenv()

# Define paths
PDF_DIR = os.path.join(os.path.dirname(__file__), "../data/slides")
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
PROCESSED_FILES_LOG = os.path.join(os.path.dirname(__file__), "../data/processed_files.json")


def get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file to track if it's been modified."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_processed_files() -> dict:
    """Load the log of processed files."""
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, 'r') as f:
            return json.load(f)
    return {}


def save_processed_files(processed_files: dict):
    """Save the log of processed files."""
    os.makedirs(os.path.dirname(PROCESSED_FILES_LOG), exist_ok=True)
    with open(PROCESSED_FILES_LOG, 'w') as f:
        json.dump(processed_files, f, indent=2)


def get_new_files(pdf_files: List[str], processed_files: dict) -> List[str]:
    """Identify files that are new or have been modified."""
    new_files = []
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        file_hash = get_file_hash(pdf_path)
        
        # Check if file is new or modified
        if filename not in processed_files or processed_files[filename]['hash'] != file_hash:
            new_files.append(pdf_path)
            print(f"   📄 New/Modified: {filename}")
        else:
            print(f"   ✓ Already processed: {filename}")
    
    return new_files


def download_pdf_from_url(url: str, save_dir: str) -> str:
    """
    Download a PDF from a URL.
    
    Args:
        url: URL of the PDF
        save_dir: Directory to save the PDF
        
    Returns:
        Path to downloaded file, or None if failed
    """
    try:
        print(f"   📥 Downloading from: {url}")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Extract filename from URL or use a default
        filename = os.path.basename(url) or "downloaded.pdf"
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        save_path = os.path.join(save_dir, filename)
        
        # Save the file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"   ✅ Downloaded: {filename}")
        return save_path
        
    except Exception as e:
        print(f"   ✗ Failed to download {url}: {e}")
        return None


def download_stanford_slp3_pdfs(save_dir: str) -> List[str]:
    """
    Attempt to download Stanford SLP3 PDFs.
    Note: This is a placeholder - actual PDF URLs need to be provided.
    The SLP3 book is available online but PDF links may vary.
    
    Args:
        save_dir: Directory to save PDFs
        
    Returns:
        List of downloaded file paths
    """
    print("\n📚 Stanford SLP3 PDF Download")
    print("   Note: Stanford SLP3 book is available at: https://web.stanford.edu/~jurafsky/slp3/")
    print("   The book is primarily web-based. You may need to:")
    print("   1. Download chapters manually from the website")
    print("   2. Use browser extensions to save pages as PDF")
    print("   3. Or provide direct PDF URLs if available")
    
    # Placeholder for PDF URLs - user should add actual URLs here
    # Example format (these are not real URLs):
    slp3_urls = [
        # "https://web.stanford.edu/~jurafsky/slp3/chapter1.pdf",
        # "https://web.stanford.edu/~jurafsky/slp3/chapter2.pdf",
        # Add more URLs as needed
    ]
    
    downloaded_files = []
    for url in slp3_urls:
        file_path = download_pdf_from_url(url, save_dir)
        if file_path:
            downloaded_files.append(file_path)
    
    return downloaded_files


def load_pdfs_with_basic_metadata(pdf_files: List[str]) -> List[Document]:
    """
    Load PDFs from the given list with basic metadata.
    
    Args:
        pdf_files: List of PDF file paths to process
        
    Returns:
        List of Document objects
    """
    if not pdf_files:
        return []

    all_documents = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n📖 Processing: {filename}")

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Add basic metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "source_file": filename,
                    "source_type": "pdf_slide",
                    "page_number": doc.metadata.get("page", i + 1),
                    "total_pages": len(documents),
                })

                # Try to extract slide title
                lines = doc.page_content.split('\n')
                if lines:
                    first_line = lines[0].strip()
                    if first_line and len(first_line) < 100:
                        doc.metadata["slide_title"] = first_line

                all_documents.append(doc)

            print(f"   ✓ Loaded {len(documents)} pages")

        except Exception as e:
            print(f"   ✗ Error loading {filename}: {e}")
            continue

    return all_documents


def create_smart_chunks(documents: List[Document]) -> List[Document]:
    """
    Create intelligent chunks that respect content structure.
    """
    print("\n✂️  Creating intelligent chunks...")

    # Smaller chunks for slides (they're already focused)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks")

    return chunks


def get_or_create_vectorstore() -> Chroma:
    """
    Get existing vectorstore or create a new one.
    """
    # Disable ChromaDB telemetry to avoid PostHog connection errors
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if os.path.exists(DB_PATH):
        print(f"   📂 Loading existing vector store from {DB_PATH}")
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_function
        )
        return vectorstore
    else:
        print(f"   📦 Creating new vector store at {DB_PATH}")
        # Create empty vectorstore
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_function
        )
        return vectorstore


def add_documents_to_vectorstore(vectorstore: Chroma, chunks: List[Document]):
    """
    Add new documents to existing vectorstore incrementally.
    Handles large batches by splitting into smaller chunks to avoid ChromaDB batch size limits.
    """
    print(f"\n➕ Adding {len(chunks)} new chunks to vector store...")
    
    # ChromaDB has a maximum batch size (typically around 5000-5500)
    # We'll use a safe batch size of 5000
    MAX_BATCH_SIZE = 5000
    import uuid
    
    # Generate unique IDs for each chunk
    ids = [f"{chunk.metadata.get('source_file', 'unknown')}_{uuid.uuid4().hex[:8]}_{i}" 
           for i, chunk in enumerate(chunks)]
    
    # Split into batches if needed
    total_chunks = len(chunks)
    if total_chunks > MAX_BATCH_SIZE:
        print(f"   ⚠️  Large batch detected ({total_chunks} chunks). Splitting into batches of {MAX_BATCH_SIZE}...")
        
        num_batches = (total_chunks + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE  # Ceiling division
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * MAX_BATCH_SIZE
            end_idx = min(start_idx + MAX_BATCH_SIZE, total_chunks)
            
            batch_chunks = chunks[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            
            print(f"   📦 Processing batch {batch_idx + 1}/{num_batches} ({len(batch_chunks)} chunks)...")
            
            # Add batch using Chroma's built-in method
            vectorstore.add_documents(
                documents=batch_chunks,
                ids=batch_ids
            )
            
            # Note: ChromaDB automatically persists when using persist_directory
            print(f"   ✅ Batch {batch_idx + 1} added successfully")
    else:
        # Small batch, add all at once
        print(f"   📦 Adding {total_chunks} chunks in single batch...")
        vectorstore.add_documents(
            documents=chunks,
            ids=ids
        )
        # Note: ChromaDB automatically persists when using persist_directory
    
    print(f"   ✅ Successfully added all {len(chunks)} chunks to vector store")


def verify_ingestion(vectorstore: Chroma):
    """
    Verify ingestion with test queries.
    """
    print("\n🔍 Verifying ingestion...")

    test_queries = [
        "What is tokenization?",
        "Explain transformers",
        "How do word embeddings work?"
    ]

    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        if results:
            print(f"\n   Query: '{query}'")
            print(f"   ✓ Found {len(results)} results")
            if results[0].metadata:
                source = results[0].metadata.get('source_file', 'unknown')
                page = results[0].metadata.get('page_number', '?')
                concept = results[0].metadata.get('concept', results[0].metadata.get('topic', 'N/A'))
                difficulty = results[0].metadata.get('difficulty', 'N/A')
                print(f"   📄 Source: {source} (page {page})")
                print(f"   🏷️  Concept: {concept} | Difficulty: {difficulty}")
        else:
            print(f"   ✗ No results for: '{query}'")


def main():
    """
    Main enhanced ingestion pipeline with incremental processing.
    """
    print("=" * 70)
    print("🚀 Enhanced NLP Course Materials Ingestion Pipeline")
    print("   (Incremental Processing Mode)")
    print("=" * 70)
    print("\nFeatures:")
    print("  • Incremental processing - only new/modified files")
    print("  • AI-powered metadata generation")
    print("  • Rich metadata: concept, explanation, example, difficulty")
    print("  • Prerequisite tracking")
    print("=" * 70)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not found!")
        print("   AI metadata generation will fail.")
        print("   Set your API key in .env file")
        response = input("\n   Continue without AI metadata? (y/N): ")
        if response.lower() != 'y':
            print("   Aborted. Please set OPENAI_API_KEY and try again.")
            return

    # Ensure directories exist
    os.makedirs(PDF_DIR, exist_ok=True)

    # Load processed files log
    processed_files = load_processed_files()
    print(f"\n📋 Processed files log: {len(processed_files)} files tracked")

    # Check for new PDFs
    all_pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if not all_pdf_files:
        print(f"\n⚠️  No PDF files found in {PDF_DIR}")
        print(f"   Please add your PDF slides to this directory")
        
        # Ask if user wants to download Stanford SLP3 PDFs
        download_choice = input("\n   Download Stanford SLP3 PDFs? (Note: You'll need to provide URLs) (y/N): ")
        if download_choice.lower() == 'y':
            download_stanford_slp3_pdfs(PDF_DIR)
            # Re-check for PDFs
            all_pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
        
        if not all_pdf_files:
            print(f"\n📋 Next steps:")
            print(f"   1. Add PDF slides to: {PDF_DIR}")
            print(f"   2. Or download Stanford SLP3 PDFs manually from: https://web.stanford.edu/~jurafsky/slp3/")
            print(f"   3. Run this script again")
            return

    print(f"\n📚 Found {len(all_pdf_files)} PDF files total")
    new_pdf_files = get_new_files(all_pdf_files, processed_files)

    if not new_pdf_files:
        print("\n✅ All files are up to date! No new files to process.")
        print("   To reprocess all files, delete the processed_files.json file")
        return

    print(f"\n🆕 Processing {len(new_pdf_files)} new/modified file(s)...")

    # Step 1: Load new PDFs
    documents = load_pdfs_with_basic_metadata(new_pdf_files)

    if not documents:
        print("\n❌ No documents loaded!")
        return

    # Step 2: Create chunks
    chunks = create_smart_chunks(documents)

    if not chunks:
        print("\n❌ No chunks created!")
        return

    # Step 3: Generate AI-powered metadata
    try:
        enhanced_chunks = batch_generate_metadata(chunks, batch_size=5)

        # Generate topic summary
        topic_summary = generate_topic_summary(enhanced_chunks)
        print(f"\n📊 Topics covered in new content: {len(topic_summary)}")
        for topic, subtopics in list(topic_summary.items())[:5]:
            print(f"   • {topic}: {len(subtopics)} subtopics")
        if len(topic_summary) > 5:
            print(f"   ... and {len(topic_summary) - 5} more topics")

    except Exception as e:
        print(f"\n⚠️  AI metadata generation failed: {e}")
        print("   Continuing with basic metadata...")
        enhanced_chunks = chunks

    # Step 4: Get or create vectorstore and add new documents
    vectorstore = get_or_create_vectorstore()
    
    # Check if this is a new vectorstore or existing one
    try:
        existing_count = vectorstore._collection.count() if hasattr(vectorstore, '_collection') else 0
        print(f"   📊 Existing chunks in vectorstore: {existing_count}")
    except:
        existing_count = 0
        print(f"   📊 Creating new vectorstore")
    
    # Add new chunks incrementally
    add_documents_to_vectorstore(vectorstore, enhanced_chunks)
    
    try:
        new_count = vectorstore._collection.count() if hasattr(vectorstore, '_collection') else len(enhanced_chunks)
    except:
        new_count = len(enhanced_chunks)
    print(f"   📊 Total chunks in vectorstore: {new_count}")

    # Step 5: Update processed files log
    for pdf_path in new_pdf_files:
        filename = os.path.basename(pdf_path)
        processed_files[filename] = {
            'hash': get_file_hash(pdf_path),
            'processed_at': str(Path(pdf_path).stat().st_mtime),
            'chunks': len([c for c in enhanced_chunks if c.metadata.get('source_file') == filename])
        }
    
    save_processed_files(processed_files)

    # Step 6: Verify
    verify_ingestion(vectorstore)

    # Success summary
    print("\n" + "=" * 70)
    print("✅ INCREMENTAL INGESTION COMPLETE!")
    print("=" * 70)
    print(f"📊 Summary:")
    print(f"   • New files processed: {len(new_pdf_files)}")
    print(f"   • New chunks added: {len(enhanced_chunks)}")
    print(f"   • Total chunks in database: {new_count}")
    print(f"   • AI-enhanced metadata: {sum(1 for c in enhanced_chunks if c.metadata.get('ai_generated', False))}")
    print(f"   • Database location: {DB_PATH}")
    print(f"\n🚀 Ready to run:")
    print(f"   See README_NEXTJS_FASTAPI.md for web interface setup")
    print("=" * 70)


if __name__ == "__main__":
    main()
