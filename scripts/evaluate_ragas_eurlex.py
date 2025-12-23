import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

from datasets import Dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness


@dataclass
class EvalConfig:
    llm_model: str
    temperature: float
    embedding_model: str
    embedding_backend: str
    retriever_k: int
    collection_name: str
    celex_id: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"

DATASET_PATH = EVAL_DIR / "eurlex_eval.json"
INGEST_CONFIG_PATH = EVAL_DIR / "eurlex_ingest_config.json"
RUN_LOG_PATH = EVAL_DIR / "eurlex_runs.jsonl"
RESULTS_JSON_PATH = EVAL_DIR / "eurlex_ragas_results.json"
RESULTS_CSV_PATH = EVAL_DIR / "eurlex_ragas_results.csv"
EVAL_CONFIG_PATH = EVAL_DIR / "eurlex_eval_config.json"

CHROMA_DIR = DATA_DIR / "chroma_db"

RETRIEVER_K = 5
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.0


def load_ingest_config() -> Dict[str, Any]:
    if INGEST_CONFIG_PATH.exists():
        return json.loads(INGEST_CONFIG_PATH.read_text())
    raise RuntimeError("Ingest config not found. Please run ingest_gdpr_eurlex.py first.")


def init_embeddings(ingest_cfg: Dict[str, Any]):
    backend = ingest_cfg.get("embedding_backend", "huggingface")
    model = ingest_cfg.get("embedding_model", "all-MiniLM-L6-v2")
    if backend == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)


def build_prompt(question: str, docs: List[Document]) -> str:
    context_blocks = []
    for idx, doc in enumerate(docs, 1):
        meta = doc.metadata
        label = meta.get("article_number", meta.get("doc_id", f"doc-{idx}"))
        title = meta.get("article_title", "")
        header = f"[{idx}] {label} {('- ' + title) if title else ''}".strip()
        context_blocks.append(f"{header}\n{doc.page_content}")

    context_text = "\n\n".join(context_blocks)
    return (
        "You are a legal assistant.\n"
        "Use only the retrieved GDPR passages to answer the question.\n"
        "Cite the passages you rely on using their bracketed numbers (e.g., [1], [2]).\n"
        "If the passages do not answer the question, say you are unsure.\n\n"
        f"Retrieved passages:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer (with citations):"
    )


def main():
    ingest_cfg = load_ingest_config()
    collection_name = ingest_cfg.get("collection_name", "eurlex_gdpr")
    embeddings = init_embeddings(ingest_cfg)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Please export it before running evaluation.")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=api_key,
        timeout=120,
        max_retries=3,
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    dataset = json.loads(DATASET_PATH.read_text())

    ragas_records: Dict[str, List[Any]] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    RUN_LOG_PATH.write_text("", encoding="utf-8")  # reset log

    for entry in dataset:
        question = entry["question"]
        reference_contexts = entry["reference_contexts"]

        # LangChain retrievers expose `invoke` in newer versions
        docs = retriever.invoke(question)

        # Persist retrieved contexts with metadata before concatenation
        retrieved_payload = []
        for doc in docs:
            md = doc.metadata or {}
            retrieved_payload.append(
                {
                    "doc_id": md.get("doc_id"),
                    "article": md.get("article_number"),
                    "article_title": md.get("article_title"),
                    "text": doc.page_content,
                }
            )

        prompt = build_prompt(question, docs)
        response = llm.invoke(prompt).content

        log_entry = {
            "question": question,
            "retrieved_contexts": retrieved_payload,
            "generated_answer": response,
            "reference_contexts": reference_contexts,
        }
        with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        ragas_records["question"].append(question)
        ragas_records["answer"].append(response)
        ragas_records["contexts"].append([doc.page_content for doc in docs])
        # RAGAS expects a string ground truth; we keep reference contexts logged separately.
        ragas_records["ground_truth"].append("\n\n".join(reference_contexts))

    ragas_ds = Dataset.from_dict(ragas_records)

    evaluation = evaluate(
        dataset=ragas_ds,
        metrics=[context_precision, context_recall, faithfulness],
        llm=llm,
        embeddings=embeddings,
    )

    eval_df = evaluation.to_pandas()
    eval_df.to_json(RESULTS_JSON_PATH, orient="records", indent=2)
    eval_df.to_csv(RESULTS_CSV_PATH, index=False)

    aggregates = {}
    for metric in ["context_precision", "context_recall", "faithfulness"]:
        raw_scores = eval_df[metric].tolist()
        scores = [
            float(s)
            for s in raw_scores
            if isinstance(s, (int, float)) and s == s  # filters out nan
        ]
        aggregates[metric] = {
            "mean": mean(scores) if scores else 0.0,
            "std": pstdev(scores) if len(scores) > 1 else 0.0,
            "count": len(scores),
            "dropped": len(raw_scores) - len(scores),
        }

    eval_config = EvalConfig(
        llm_model=LLM_MODEL,
        temperature=TEMPERATURE,
        embedding_model=ingest_cfg.get("embedding_model"),
        embedding_backend=ingest_cfg.get("embedding_backend"),
        retriever_k=RETRIEVER_K,
        collection_name=collection_name,
        celex_id=ingest_cfg.get("celex_id"),
    )

    EVAL_CONFIG_PATH.write_text(json.dumps(asdict(eval_config), indent=2), encoding="utf-8")

    summary_payload = {
        "per_question_path": str(RESULTS_JSON_PATH),
        "aggregate": aggregates,
        "config_path": str(EVAL_CONFIG_PATH),
    }
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
