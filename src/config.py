# src/config.py
from dataclasses import dataclass
from pathlib import Path


# ✅ 永远以“项目根目录”为准：.../studycopilot-v2/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RAGConfig:
    # ============================
    # Paths (ABSOLUTE)
    # ============================
    data_dir: Path = PROJECT_ROOT / "data"
    vectordb_dir: Path = PROJECT_ROOT / "vectordb"

    # ============================
    # Models
    # ============================
    llm_model: str = "llama3"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ============================
    # Retrieval
    # ============================
    candidate_k: int = 8
    top_k: int = 4

    enable_hybrid: bool = True
    bm25_k: int = 5

    enable_query_router: bool = True
    n_rewrites: int = 2

    # Evidence gate
    min_evidence_chars: int = 120

    # ReAct
    react_max_steps: int = 3

    # Debug
    show_debug: bool = True