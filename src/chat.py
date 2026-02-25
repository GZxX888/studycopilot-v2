# src/chat.py
from __future__ import annotations

from typing import List, Tuple
import re
from difflib import SequenceMatcher

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi

from config import RAGConfig

# ✅ NEW: rewrite + refine
from query_rewrite import build_query_rewrite_chain
from query_refine import build_query_refine_chain


# ============================================================
# 1) 去重
# ============================================================

def remove_similar_docs(docs: List[Document], threshold: float = 0.9) -> List[Document]:
    filtered = []
    for d in docs:
        text = d.page_content.strip()
        is_dup = False
        for kept in filtered:
            ratio = SequenceMatcher(None, text, kept.page_content.strip()).ratio()
            if ratio > threshold:
                is_dup = True
                break
        if not is_dup:
            filtered.append(d)
    return filtered


# ============================================================
# 2) Query Router (complexity)
# ============================================================

def question_complexity_level(question: str) -> int:
    q = " " + question.lower().strip() + " "

    multi_markers = [
        " and ", " or ", " compare ", " difference ",
        " advantages ", " disadvantages ", " steps ",
        " derive ", " prove "
    ]

    long_question = len(q.split()) > 25
    multi_hit = any(m in q for m in multi_markers)

    if multi_hit and long_question:
        return 2
    if multi_hit or long_question:
        return 1
    return 0


# ============================================================
# 3) Hybrid Retrieval Core
# ============================================================

def build_strong_retriever(cfg: RAGConfig):
    """
    对外暴露的强检索工具
    return:
        strong_retrieve(query) ->
            docs_for_llm, queries_used, debug_notes
    """

    # === 向量库加载 ===
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        persist_directory=str(cfg.vectordb_dir),
        embedding_function=embeddings,
        collection_name="studycopilot",
    )

    dense_retriever = vectordb.as_retriever(
        search_kwargs={"k": cfg.candidate_k}
    )

    # === BM25 构建 ===
    store_dump = vectordb.get()
    all_docs = store_dump.get("documents") or []
    all_metadatas = store_dump.get("metadatas") or []

    # 空库保护：避免 BM25 division by zero
    if len(all_docs) == 0:
        raise RuntimeError(
            "VectorDB is empty (no documents found). "
            "Please ingest/rebuild your knowledge base first (run ingest or click 'Rebuild Knowledge Base' in Streamlit)."
        )

    tokenized_corpus = [
        re.findall(r"\w+", doc.lower())
        for doc in all_docs
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    llm = OllamaLLM(model=cfg.llm_model, temperature=0)

    # ✅ NEW: build rewrite + refine chains
    rewrite_chain = build_query_rewrite_chain(llm)
    refine_chain, refine_parse = build_query_refine_chain(
        llm,
        n_candidates=getattr(cfg, "n_refine_candidates", 4)
    )

    ranker = Ranker()

    final_k = cfg.top_k
    enable_router = getattr(cfg, "enable_query_router", True)
    enable_hybrid = getattr(cfg, "enable_hybrid", True)

    # 可选：允许你用 config 开关 refine（默认开）
    enable_refine = getattr(cfg, "enable_query_refine", True)

    # --------------------------------------------------------
    # 核心函数
    # --------------------------------------------------------

    def strong_retrieve(query: str) -> Tuple[List[Document], List[str], List[str]]:

        debug_notes: List[str] = []

        # ---- Router ----
        if enable_router:
            level = question_complexity_level(query)
        else:
            level = 1

        # =====================================================
        # ✅ Rewrite + Refinement（替换你原来的 generate_rewrites）
        # =====================================================

        rewritten_query = ""
        best_query = query
        candidates: List[str] = []

        if enable_refine and level >= 1:
            # 1) Rewrite（1条）
            try:
                rewritten_query = (rewrite_chain.invoke({"question": query}) or "").strip()
            except Exception as e:
                debug_notes.append(f"REWRITE_ERROR={type(e).__name__}")

            # 2) Refinement（多候选 + best）
            try:
                refine_raw = refine_chain.invoke({
                    "question": query,
                    "rewritten_query": rewritten_query
                })
                refine_data = refine_parse(refine_raw)
                best_query = (refine_data.get("best_query") or "").strip()
                candidates = refine_data.get("candidates") or []
            except Exception as e:
                debug_notes.append(f"REFINE_ERROR={type(e).__name__}")

            # best_query兜底
            if not best_query:
                best_query = rewritten_query if rewritten_query else query

        # queries_used：把 best 放第一，其它作为扩展检索
        queries: List[str] = []
        queries.append(best_query)

        # candidates（可为空）
        for c in candidates:
            if isinstance(c, str) and c.strip():
                queries.append(c.strip())

        # 额外兜底：把 rewritten / original 放进去（避免 refine 太激进）
        if rewritten_query and rewritten_query.strip():
            queries.append(rewritten_query.strip())
        queries.append(query)

        # 去重 queries（保持顺序）
        seen_q = set()
        deduped_queries = []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen_q:
                seen_q.add(key)
                deduped_queries.append(q.strip())
        queries = deduped_queries

        debug_notes.append(f"LEVEL={level}")
        debug_notes.append(f"REWRITTEN={rewritten_query}")
        debug_notes.append(f"BEST_QUERY={best_query}")
        debug_notes.append(f"QUERIES={queries}")

        # ---- Dense + Sparse 合并 ----
        merged_docs: List[Document] = []
        seen = set()

        for q in queries:

            # Dense
            dense_docs = dense_retriever.invoke(q)
            for d in dense_docs:
                key = (d.metadata.get("source"),
                       d.metadata.get("page"),
                       d.page_content[:120])
                if key not in seen:
                    seen.add(key)
                    merged_docs.append(d)

            # Sparse (BM25)
            if enable_hybrid:
                tokenized_query = re.findall(r"\w+", q.lower())
                scores = bm25.get_scores(tokenized_query)

                top_indices = sorted(
                    range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True
                )[: cfg.bm25_k]

                for idx in top_indices:
                    doc_text = all_docs[idx]
                    metadata = all_metadatas[idx]
                    fake_doc = Document(page_content=doc_text, metadata=metadata)

                    key = (metadata.get("source"),
                           metadata.get("page"),
                           doc_text[:120])
                    if key not in seen:
                        seen.add(key)
                        merged_docs.append(fake_doc)

        # ---- 去重 ----
        merged_docs = remove_similar_docs(merged_docs)

        # ---- Rerank ----
        if merged_docs:
            passages = [
                {"id": str(i), "text": d.page_content}
                for i, d in enumerate(merged_docs)
            ]
            request = RerankRequest(query=query, passages=passages)
            ranked = ranker.rerank(request)

            top_ids = [int(r["id"]) for r in ranked[: max(final_k, 1)]]
            docs_for_llm = [
                merged_docs[i] for i in top_ids
                if 0 <= i < len(merged_docs)
            ]
        else:
            docs_for_llm = []

        debug_notes.append(f"DOCS_AFTER_RERANK={len(docs_for_llm)}")

        return docs_for_llm, queries, debug_notes

    return strong_retrieve