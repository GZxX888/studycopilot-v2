
from __future__ import annotations

# ===== 标准库 =====
from typing import List, Dict, Any

# ===== LangChain =====
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== 项目内 =====
from config import RAGConfig
from loaders import load_documents_from_dir


# ============================================================
# 1) 低信息过滤
# ============================================================
def is_low_information_chunk(text: str) -> bool:
    if not text:
        return True

    t = text.strip().lower()

    if len(t) < 30:
        return True

    blacklist = [
        "thanks",
        "thank you",
        "follow the slope",
        "questions?",
        "any questions",
        "end of lecture",
    ]
    for b in blacklist:
        if b in t:
            return True

    letters = sum(c.isalpha() for c in t)
    if letters / max(len(t), 1) < 0.3:
        return True

    # PPT 渐进残留页常见模式
    if t.count("what is") > 3:
        return True

    return False


# ============================================================
# 2) 移除渐进式幻灯片页（保留“最后一页”）
#    逻辑：如果当前页是上一页的“子串”，说明它更短、更早，是过渡页 → 删
# ============================================================
def remove_progressive_slides(docs: List[Document]) -> List[Document]:
    cleaned: List[Document] = []
    prev_text = ""

    for doc in docs:
        text = doc.page_content.strip()
        if len(text) < 20:
            continue

        # 当前页是上一页的子串 => 当前更短，是过渡页，丢弃
        if prev_text and text in prev_text:
            continue

        cleaned.append(doc)
        prev_text = text

    return cleaned


# ============================================================
# 3) 去重（完全相同文本）
# ============================================================
def deduplicate_docs(docs: List[Document]) -> List[Document]:
    unique: List[Document] = []
    seen = set()
    for d in docs:
        t = d.page_content.strip()
        if t in seen:
            continue
        seen.add(t)
        unique.append(d)
    return unique


# ============================================================
# 4) semantic chunking（页内语义分块）
#    - PPT 通常“一页就是一个语义单元”，但有些页很长（例题、推导）会太大。
#    - 策略：只对“超长页”做页内拆分；短页保持整页。
# ============================================================
def semantic_chunk_pages(
    docs: List[Document],
    max_page_chars: int,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )

    out: List[Document] = []
    for d in docs:
        text = d.page_content.strip()

        # 短页直接保留，不拆
        if len(text) <= max_page_chars:
            out.append(d)
            continue

        # 长页才拆分（页内语义拆）
        sub_docs = splitter.split_documents([d])
        out.extend(sub_docs)

    return out


def ingest_documents(cfg: RAGConfig) -> Dict[str, Any]:
    """
    Ingest pipeline:
    - load docs from cfg.data_dir
    - clean/filter/dedup
    - semantic chunking
    - embed + persist to Chroma (collection_name="studycopilot")

    Returns stats for UI / logging.
    """
    # ---------- 1) load ----------
    docs: List[Document] = load_documents_from_dir(cfg.data_dir)
    if not docs:
        raise RuntimeError("No documents found in data directory.")

    original_pages = len(docs)

    # ---------- 2) remove progressive ----------
    docs = remove_progressive_slides(docs)
    after_progressive = len(docs)

    # ---------- 3) low-info filter (page-level) ----------
    docs = [d for d in docs if not is_low_information_chunk(d.page_content)]
    after_low_info = len(docs)

    # ---------- 4) dedup ----------
    docs = deduplicate_docs(docs)
    after_dedup = len(docs)

    # ---------- 5) semantic chunking (page-internal) ----------
    max_page_chars = 1800
    chunk_size = getattr(cfg, "chunk_size", 800)
    chunk_overlap = getattr(cfg, "chunk_overlap", 120)

    chunks = semantic_chunk_pages(
        docs,
        max_page_chars=max_page_chars,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    final_chunks = len(chunks)

    # ---------- 6) Embeddings (normalize) ----------
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    # ---------- 7) VectorDB ----------
    cfg.vectordb_dir.mkdir(parents=True, exist_ok=True)
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(cfg.vectordb_dir),
        collection_name="studycopilot",
    )

    return {
        "original_pages": original_pages,
        "after_progressive": after_progressive,
        "after_low_information": after_low_info,
        "after_deduplication": after_dedup,
        "after_semantic_chunking": final_chunks,
        "final_chunks": final_chunks,
        "vectordb_path": str(cfg.vectordb_dir),
        "collection": "studycopilot",
    }


def run_ingest(cfg: RAGConfig | None = None) -> Dict[str, Any]:
    """
    Streamlit-callable entry.
    Returns stats dict.
    """
    cfg = cfg or RAGConfig()
    stats = ingest_documents(cfg)
    return stats


if __name__ == "__main__":
    out = run_ingest()
    print(f"Original pages: {out['original_pages']}")
    print(f"After removing progressive slides: {out['after_progressive']}")
    print(f"After low-information filtering: {out['after_low_information']}")
    print(f"After deduplication: {out['after_deduplication']}")
    print(f"After semantic chunking: {out['after_semantic_chunking']}")
    print("\n✅ Ingest finished")
    print(f"- Final Chunks: {out['final_chunks']}")
    print(f"- VectorDB path: {out['vectordb_path']}")