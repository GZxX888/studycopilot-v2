# src/agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

from config import RAGConfig
from rag_core import format_docs
from chat import build_strong_retriever  # 从 chat.py 暴露的强检索工具


FALLBACK_PROMPT = """You are a helpful assistant.

The user's question may be outside the provided course notes.

Rules:
- Detect the language of the QUESTION.
- Answer strictly in that language.
- If the user asks about course-specific facts (e.g., specific slide/page numbers, what the notes say), say you are not sure without the notes.
- Otherwise you may answer using general knowledge.
- Do NOT invent citations.
- Be concise.

QUESTION:
{question}

Answer:
"""

ROUTER_PROMPT = """You are a routing controller for StudyCopilot.

Your job is to decide whether a question requires retrieving from lecture notes (RAG) or not (NO_RAG).

Use these rules strictly:

Return RAG if:
- The question is about course concepts.
- It references lectures, slides, pages, assignments.
- It asks for definitions or explanations likely in the notes.

Return NO_RAG if:
- The question is general world knowledge.
- It is common knowledge (e.g., capital cities, math facts).
- It does not depend on the specific course notes.

Return ONLY one token:
RAG
NO_RAG

User question:
{question}
"""

EVIDENCE_CHECK_PROMPT = """You are an evidence gate.
Given QUESTION and CONTEXT, decide if the context contains sufficient evidence to answer.

Return ONLY one token:
- ENOUGH
- NOT_ENOUGH

Be strict.

QUESTION:
{question}

CONTEXT:
{context}
"""

ANSWER_PROMPT = """You are StudyCopilot.

Use ONLY the CONTEXT for factual information.

Rules:
- Detect the language of the QUESTION.
- The ANSWER must follow the language of the QUESTION.
- You MAY translate information from the CONTEXT into the language of the QUESTION.
- Do NOT mix languages.
- Cite like [1], [2]. Do NOT invent citations.
- If insufficient, say you don't have enough evidence (in the language of the QUESTION) and stop.

Structure:
1. Short Definition
2. Intuition
3. Formula (if relevant)
4. Connection to neural networks

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""


# ✅ ReAct 只用来“决定下一步怎么检索/是否结束”，最终回答仍用 ANSWER_PROMPT_* 生成
REACT_PLANNER_PROMPT = """You are a ReAct-style planner for a study assistant.

Goal:
- Decide whether to retrieve more evidence from course notes, or stop and answer.

You have ONE tool:
- retrieve[query]  -> returns lecture excerpts

Rules:
- If you still lack evidence, you MUST call retrieve[...] again with a better query.
- If evidence looks sufficient, choose FINAL.

Output format (STRICT):
Thought: <one short sentence>
Action: retrieve[...]
or
Thought: <one short sentence>
Action: FINAL

User question:
{question}

Current evidence summary (may be empty):
{evidence_summary}
"""


@dataclass
class AgentState:
    route: str
    docs_found: int
    citations: List[str]


class StudyCopilotAgent:
    def _wants_bilingual(self, question: str) -> bool:
        q = question.lower()
        return (
            ("both" in q and "chinese" in q and "english" in q)
            or ("中英" in q)
            or ("中文" in q and "英文" in q)
        )

    def _is_chinese(self, text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

        # 用 OllamaLLM（你原来就是这个）
        self.llm = OllamaLLM(model=cfg.llm_model, temperature=0.2)

        # ✅ 路线2：用 chat.py 的强检索器（hybrid + router + rewrite + rerank）
        # build_strong_retriever(cfg) -> callable(query)-> (docs_for_llm, queries_used, debug_notes)
        self.strong_retrieve = build_strong_retriever(cfg)

        # ReAct 参数（没有就给默认值，避免你 config 不完整）
        self.react_max_steps = getattr(cfg, "react_max_steps", 3)

    # -----------------------------
    # Router / Evidence / Fallback
    # -----------------------------
    def _route(self, question: str) -> str:
        if not getattr(self.cfg, "enable_query_router", True):
            return "RAG"

        resp = self.llm.invoke(ROUTER_PROMPT.format(question=question)).strip().upper()
        if "NO_RAG" in resp:
            return "NO_RAG"
        if "RAG" in resp:
            return "RAG"
        return "RAG"

    def _evidence_gate(self, question: str, context: str) -> bool:
        # 先用硬阈值挡一层
        min_chars = getattr(self.cfg, "min_evidence_chars", 200)
        if len(context) < min_chars:
            return False

        resp = self.llm.invoke(
            EVIDENCE_CHECK_PROMPT.format(question=question, context=context)
        ).strip().upper()
        return resp.startswith("ENOUGH")

    def _fallback_answer(self, question: str) -> str:
        prompt = FALLBACK_PROMPT.format(question=question)
        resp = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm.predict(prompt)
        return (resp or "").strip()

    # -----------------------------
    # ReAct helpers
    # -----------------------------
    def _parse_planner_action(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Return: ("retrieve", query) | ("final", None) | ("unknown", None)
        """
        t = (text or "").strip()

        # Action: FINAL
        if re.search(r"Action:\s*FINAL\b", t, flags=re.IGNORECASE):
            return ("final", None)

        # Action: retrieve[...]
        m = re.search(r"Action:\s*retrieve\[(.+?)\]\s*$", t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            q = m.group(1).strip()
            if q:
                return ("retrieve", q)

        return ("unknown", None)

    def _make_evidence_summary(self, docs: List[Document], max_chars_per_doc: int = 180) -> str:
        """
        给 planner 看一个“轻量摘要”，避免把整个 context 塞回去造成 prompt 爆炸。
        """
        if not docs:
            return ""

        lines = []
        for i, d in enumerate(docs[: min(4, len(docs))], start=1):
            snippet = (d.page_content or "").strip().replace("\n", " ")
            if len(snippet) > max_chars_per_doc:
                snippet = snippet[:max_chars_per_doc] + "..."
            src = d.metadata.get("source", "")
            page = d.metadata.get("page", None)
            ref = f"{src} p{page}" if page is not None else src
            lines.append(f"[{i}] {ref} :: {snippet}")
        return "\n".join(lines)

    def _react_retrieve_loop(self, question: str) -> Tuple[List[Document], List[str], List[str]]:
        """
        ReAct loop 只负责：多轮决定是否继续检索 / 用什么 query 检索
        最终输出：best_docs, all_queries_used, debug_notes
        """
        best_docs: List[Document] = []
        all_queries: List[str] = []
        debug_notes: List[str] = []

        seen_norm_queries: List[str] = []
        evidence_summary = ""

        # 初始：让 planner 决定要不要先 retrieve（大多数会 retrieve）
        for step in range(1, self.react_max_steps + 1):
            planner_prompt = REACT_PLANNER_PROMPT.format(
                question=question,
                evidence_summary=evidence_summary or "(empty)"
            )

            planner_out = self.llm.invoke(planner_prompt).strip()
            action, payload = self._parse_planner_action(planner_out)

            debug_notes.append(f"[planner step {step}] {planner_out}")

            if action == "final":
                # planner 认为证据够了，结束检索
                break

            if action != "retrieve" or not payload:
                # 容错：planner 没按格式输出，直接用原问题检索一次
                payload = question
                debug_notes.append(f"[planner step {step}] malformed action, fallback retrieve with original question.")

            #  Step1: repeated query detector
            norm_q = re.sub(r"\s+", " ", (payload or "").strip().lower())

            if any(norm_q == q for q in seen_norm_queries):
                debug_notes.append(
                    f"[planner step {step}] repeated query detected. Triggering rewrite."
                )

                # 🔥 自动改写 query
                rewrite_prompt = f"""
            The previous retrieval query did not provide sufficient evidence.

            Rewrite the query to be more specific and include related technical terms.

            Original question:
            {question}

            Previous query:
            {payload}

            Provide ONE improved retrieval query only.
            """
                new_query = self.llm.invoke(rewrite_prompt).strip()

                debug_notes.append(f"[planner step {step}] rewritten query = {new_query}")

                norm_new = re.sub(r"\s+", " ", new_query.lower().strip())

                # 如果改写后还是一样，才真正停止
                if norm_new in seen_norm_queries:
                    debug_notes.append(
                        f"[planner step {step}] rewrite still same. Stopping loop."
                    )
                    break

                seen_norm_queries.append(norm_new)
                payload = new_query
            else:
                seen_norm_queries.append(norm_q)

            seen_norm_queries.append(norm_q)
            # ✅ 强检索（chat.py）
            docs, queries_used, dbg = self.strong_retrieve(payload)
            all_queries.extend(queries_used or [payload])
            debug_notes.extend(dbg or [])

            # 更新 best_docs（简单策略：谁更长/更多就保留）
            # 也可以改成“按 evidence gate 通过优先”
            if len(docs) > len(best_docs):
                best_docs = docs

            evidence_summary = self._make_evidence_summary(best_docs)

        return best_docs, all_queries, debug_notes

    # -----------------------------
    # Public API
    # -----------------------------
    def answer(self, question: str) -> Dict[str, Any]:
        route = self._route(question)

        # 1) NO_RAG：保持原逻辑
        if route == "NO_RAG":
            final = self.llm.invoke(
                "Answer briefly and safely. If unsure, say you are unsure.\n\nQuestion:\n" + question
            ).strip()
            final = "⚠️ **Answered without note evidence (NO_RAG).**\n\n" + final
            return {
                "final": final,
                "route": route,
                "citations": [],
                "docs_found": 0,
                "fallback": True,
                "debug": [],
            }

        # 2) RAG：走 ReAct（路线2）
        docs, queries_used, debug_notes = self._react_retrieve_loop(question)
        context, citations = format_docs(docs)

        if not context:
            fb = self._fallback_answer(question)
            msg = "**在你的笔记里没有找到相关证据。**" if self._is_chinese(
                question) else "**No relevant evidence found in your notes.**"
            final = msg + "\n\n" + fb
            return {
                "final": final,
                "route": route,
                "citations": [],
                "docs_found": 0,
                "fallback": True,
                "debug": debug_notes,
                "queries_used": queries_used,
            }

        if not self._evidence_gate(question, context):
            fb = self._fallback_answer(question)
            msg = "**证据不足：我在你的笔记里找不到足够内容来可靠回答。**" if self._is_chinese(
                question) else "**Not enough evidence in your notes to answer confidently.**"
            final = msg + "\n\n" + fb
            return {
                "final": final,
                "route": route,
                "citations": citations,  # 可以展示检索到什么，但不把它当证据
                "docs_found": len(docs),
                "fallback": True,
                "debug": debug_notes,
                "queries_used": queries_used,
            }

        prompt_tpl = ANSWER_PROMPT
        prompt = prompt_tpl.format(context=context, question=question)

        resp = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm.predict(prompt)
        final = (resp or "").strip()

        return {
            "final": final,
            "route": route,
            "citations": citations,
            "docs_found": len(docs),
            "fallback": False,
            "debug": debug_notes,
            "queries_used": queries_used,
        }