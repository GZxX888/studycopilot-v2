# src/config.py
from dataclasses import dataclass
from pathlib import Path


# 永远以「项目根目录」为准
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
    quick_retrieve_k: int = 2  # 轻量预检索数量

    enable_hybrid: bool = True
    bm25_k: int = 5

    enable_query_router: bool = True
    n_rewrites: int = 2

    # ============================
    # Evidence Gate
    # ============================
    min_evidence_chars: int = 100  # 降低阈值，减少误拒绝
    evidence_score_threshold: int = 40  # 证据评分阈值 (0-100)
    verification_score_threshold: int = 70  # 答案验证阈值 (0-100)

    # ============================
    # Query Decomposition
    # ============================
    # ⚠️ 暂时关闭：分解出的子问题各跑一轮完整 ReAct（含 rewrite+refine），
    #    成本约 3 倍，但最终 context 被 top_k*2 截断，后续子问题的证据基本进不去，
    #    实测为负收益。代码保留，修好证据配额后可一行开回。
    enable_decomposition: bool = False
    max_sub_questions: int = 3

    # ============================
    # Self-Verification
    # ============================
    # ⚠️ 暂时关闭：llama3 几乎不输出纯 JSON，_verify_answer 解析失败会返回 70，
    #    而阈值恰好也是 70（判断是 < 70）→ 该门恒为通过，只花钱不生效。
    #    换 format="json" 或换模型解决稳定性后可一行开回。
    enable_verification: bool = False
    max_refine_attempts: int = 1  # 验证失败后最多重试次数

    # ============================
    # ReAct
    # ============================
    react_max_steps: int = 3

    # ============================
    # Debug
    # ============================
    show_debug: bool = True