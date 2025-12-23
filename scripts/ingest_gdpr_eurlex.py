import json
import os
import re
import time
from pathlib import Path
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None  # type: ignore


BASE_URL = "https://www.privacy-regulation.eu/en/"
INDEX_URL = f"{BASE_URL}index.htm"

# Storage locations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = DATA_DIR / "gdpr_index.html"
ARTICLES_CACHE_DIR = DATA_DIR / "gdpr_articles"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "eurlex_gdpr"
CONFIG_PATH = DATA_DIR / "eval" / "eurlex_ingest_config.json"

# Metadata constants
CELEX_ID = "32016R0679"
SOURCE_NAME = "EUR-Lex"


def fetch_url(url: str, dest: Path) -> None:
    """Download URL to destination with basic retry."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and resp.text:
            dest.write_text(resp.text, encoding="utf-8")
            return
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after retries (status {resp.status_code})")


def load_index_html() -> str:
    """Ensure index HTML is available locally and return its content."""
    if INDEX_PATH.exists():
        return INDEX_PATH.read_text(encoding="utf-8", errors="ignore")
    fetch_url(INDEX_URL, INDEX_PATH)
    return INDEX_PATH.read_text(encoding="utf-8", errors="ignore")


def parse_article_links(index_html: str) -> List[Tuple[str, str, str]]:
    """
    Parse article links from the index HTML.

    Returns list of (article_number, article_title, absolute_url).
    """
    soup = BeautifulSoup(index_html, "html.parser")
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("article-") or not href.endswith(".htm"):
            continue
        number_match = re.search(r"article-(\d+)", href)
        if not number_match:
            continue
        article_number = f"Article {number_match.group(1)}"
        # Title typically adjacent in the same anchor text or nearby
        title_text = a.get_text(strip=True)
        if title_text.lower().startswith("article"):
            # Look ahead for a sibling anchor text that holds the title
            next_text = a.find_next(string=True)
            title_text = (
                next_text.strip(' "')
                if isinstance(next_text, str) and next_text.strip()
                else title_text
            )
        key = (article_number, href)
        if key in seen:
            continue
        seen.add(key)
        links.append(
            (
                article_number,
                title_text.strip(' "'),
                BASE_URL + href,
            )
        )
    return links


def fetch_article(url: str) -> str:
    """Fetch article page HTML, caching to disk."""
    name = url.split("/")[-1]
    dest = ARTICLES_CACHE_DIR / name
    if dest.exists():
        return dest.read_text(encoding="utf-8", errors="ignore")
    fetch_url(url, dest)
    return dest.read_text(encoding="utf-8", errors="ignore")


def extract_article_content(article_html: str) -> Tuple[str, str]:
    """
    Extract (title, text) from an article HTML page.

    The pages are table-based; we rely on text parsing:
    - Title: first occurrence of 'Article X ...' line
    - Content: lines starting at the first numbered paragraph (e.g., '1.')
    """
    soup = BeautifulSoup(article_html, "html.parser")
    full_text = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]

    # Title extraction
    title = ""
    art_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"Article\s+\d+", ln):
            art_idx = i
            # Next meaningful line usually carries the descriptive title
            for lookahead in lines[i + 1 : i + 5]:
                if lookahead and not lookahead.startswith("EU GDPR"):
                    title = lookahead.strip(' "')
                    break
            if not title:
                title = ln
            break

    # Content extraction: start from first numbered paragraph like "1."
    start_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^\d+\.", ln):
            start_idx = i
            break
    if start_idx is None:
        # Fallback: take everything after the article line
        start_idx = art_idx + 1 if art_idx is not None else 0

    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        if "Return to the top" in lines[j]:
            end_idx = j
            break

    content = " ".join(lines[start_idx:end_idx]).strip()
    return title, content


def init_embeddings():
    """Try HuggingFace embeddings first; fallback to OpenAI embeddings."""
    hf_url = "https://huggingface.co"
    hf_reachable = False
    try:
        requests.get(hf_url, timeout=5)
        hf_reachable = True
    except Exception:
        hf_reachable = False

    if hf_reachable:
        try:
            return (
                HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                ),
                "all-MiniLM-L6-v2",
                "huggingface",
            )
        except Exception as hf_error:
            print(f"⚠️ HuggingFace embedding load failed, falling back to OpenAI: {hf_error}")

    try:
        if OpenAIEmbeddings is None:
            raise RuntimeError("OpenAI embeddings not installed and HuggingFace unavailable.")
        model_name = "text-embedding-3-small"
        return OpenAIEmbeddings(model=model_name), model_name, "openai"
    except Exception as oe_error:
        raise RuntimeError(f"Failed to initialize any embedding model: {oe_error}")


def ingest_articles(docs: List[Document], embeddings, collection_name: str) -> None:
    """Persist documents into Chroma under the given collection name."""
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )


def main():
    index_html = load_index_html()
    links = parse_article_links(index_html)
    if not links:
        raise RuntimeError("No article links found in GDPR index; cannot proceed.")

    embeddings, embedding_model_name, embedding_backend = init_embeddings()

    documents: List[Document] = []
    for article_number, _, url in links:
        html = fetch_article(url)
        article_title, content = extract_article_content(html)
        if not content:
            continue
        doc_id = url.split("/")[-1].replace(".htm", "")
        metadata = {
            "source": SOURCE_NAME,
            "celex_id": CELEX_ID,
            "article_number": article_number,
            "article_title": article_title,
            "url": url,
            "doc_id": doc_id,
        }
        documents.append(Document(page_content=content, metadata=metadata))

    if not documents:
        raise RuntimeError("No documents extracted from GDPR articles.")

    ingest_articles(documents, embeddings, COLLECTION_NAME)

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "collection_name": COLLECTION_NAME,
                "embedding_model": embedding_model_name,
                "embedding_backend": embedding_backend,
                "source": SOURCE_NAME,
                "celex_id": CELEX_ID,
                "documents_ingested": len(documents),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Ingested {len(documents)} GDPR articles into collection '{COLLECTION_NAME}'.")
    print(f"Embedding model: {embedding_backend}:{embedding_model_name}")
    print(f"Config saved to: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
