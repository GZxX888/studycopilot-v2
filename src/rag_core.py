# src/rag_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document


@dataclass
class RAGConfig:
    persist_dir: str
    embedding_model: str
    top_k: int = 4


def build_retriever(cfg: RAGConfig):
    """
    Load existing Chroma vectordb (persisted) and return a retriever.
    IMPORTANT: embedding_model MUST match the one used during ingest.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        encode_kwargs={"normalize_embeddings": True},  # match your ingest
    )

    vectordb = Chroma(
        persist_directory=cfg.persist_dir,
        embedding_function=embeddings,
        collection_name="studycopilot",  # 固定读有数据的 collection
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": cfg.top_k})
    return retriever



def format_docs(docs: List[Document]) -> Tuple[str, List[str]]:
    """
    Returns:
        context_str: 拼接后的文本
        citations:   干净的 citation 列表
    """

    context_parts = []
    citations = []

    for i, doc in enumerate(docs, start=1):
        text = doc.page_content.strip()

        source_path = doc.metadata.get("source", "")
        page = doc.metadata.get("page", None)

        filename = Path(source_path).name if source_path else "Unknown"

        # 构造干净 citation
        if page is not None:
            citation_str = f"[{i}] {filename} (page {page})"
        else:
            citation_str = f"[{i}] {filename}"

        citations.append(citation_str)

        # 给模型用的 context
        context_parts.append(f"[{i}]\n{text}")

    context_str = "\n\n".join(context_parts)

    return context_str, citations