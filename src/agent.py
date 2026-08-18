# src/agent.py
"""
StudyCopilot Agent — 主流程

核心改进：
1. Post-Retrieval Routing - 先检索再决定路由
2. 渐进式三层 Gate - 早失败早退出
3. Query Decomposition - 复杂问题分解
4. Self-Verification - 答案自验证
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

from config import RAGConfig
from rag_core import format_docs
from chat import build_strong_retriever, build_quick_retriever


# ============================================================
# Prompts
# ============================================================

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

# Post-Retrieval Router: 基于预检索证据决定路由
POST_RETRIEVAL_ROUTER_PROMPT = """You are a routing controller for StudyCopilot.

You have already retrieved some evidence from the lecture notes.
Based on the evidence quality, decide the best routing strategy.

QUESTION:
{question}

RETRIEVED EVIDENCE PREVIEW:
{evidence_preview}

Choose ONE routing strategy:

- RAG: The evidence is related to the question. Proceed with the RAG pipeline.
- NO_RAG: The evidence is completely irrelevant. Answer with general knowledge.

Return ONLY one token: RAG or NO_RAG
"""

# Query Decomposition: 复杂问题分解
DECOMPOSE_PROMPT = """Analyze if this question needs decomposition into sub-questions.

QUESTION:
{question}

Rules:
- If the question asks about multiple concepts, comparisons, or has multiple parts, decompose it.
- If it's a simple single-concept question, return the original question only.
- Maximum {max_sub} sub-questions.
- Each sub-question should be self-contained and answerable independently.

Output STRICT JSON:
{{"needs_decomposition": true/false, "sub_questions": ["q1", "q2", ...]}}

If no decomposition needed:
{{"needs_decomposition": false, "sub_questions": ["{question}"]}}
"""

# Evidence Scoring: 证据评分 (Gate 2)
EVIDENCE_SCORE_PROMPT = """You are an evidence evaluator.

Given the QUESTION and retrieved CONTEXT, score the evidence quality from 0 to 100.

Consider:
- Relevance: Does the context address the question topic?
- Completeness: Does it contain enough information to answer?
- Specificity: Is the information specific or too general?

QUESTION:
{question}

CONTEXT:
{context}

Output ONLY a number from 0-100 representing the evidence score.
Example: 75
"""

# Answer Generation
ANSWER_PROMPT = """You are StudyCopilot.

Use ONLY the CONTEXT for factual information.

Rules:
- Detect the language of the QUESTION.
- The ANSWER must follow the language of the QUESTION.
- You MAY translate information from the CONTEXT into the language of the QUESTION.
- Do NOT mix languages.
- Cite like [1], [2]. Do NOT invent citations.
- If insufficient evidence, say you don't have enough evidence and stop.

Structure your answer clearly:
1. Direct answer to the question
2. Supporting details from context
3. Relevant formulas or examples (if applicable)

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""

# Self-Verification: 答案自验证 (Gate 3)
VERIFY_PROMPT = """Review this answer for faithfulness to the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Evaluate:
1. Citation Coverage: Does every factual claim have a citation? (0-100)
2. Citation Accuracy: Are citations accurate to the context? (0-100)
3. Completeness: Are important points from context covered? (0-100)
4. No Hallucination: Is everything in the answer supported by context? (0-100)

Output STRICT JSON:
{{
    "citation_coverage": <0-100>,
    "citation_accuracy": <0-100>,
    "completeness": <0-100>,
    "no_hallucination": <0-100>,
    "overall_score": <0-100>,
    "issues": ["list of specific issues found"]
}}
"""

# Answer Refinement: 答案修复
REFINE_PROMPT = """The previous answer had issues. Please fix them.

CONTEXT:
{context}

QUESTION:
{question}

PREVIOUS ANSWER:
{previous_answer}

ISSUES FOUND:
{issues}

Generate an improved answer that:
1. Addresses all the issues listed
2. Only uses information from CONTEXT
3. Properly cites sources like [1], [2]
4. Follows the language of the QUESTION

Improved Answer:
"""

# ReAct Planner (扩展版)
REACT_PLANNER_PROMPT = """You are a ReAct-style planner for a study assistant.

Goal: Decide the next action to gather sufficient evidence.

Available Actions:
- retrieve[query]: Search lecture notes with a specific query
- FINAL: Stop retrieval, evidence is sufficient

Rules:
- If evidence is empty or insufficient, you MUST call retrieve[...] with a specific query.
- If evidence looks sufficient to answer the question, choose FINAL.
- Be specific in your retrieve queries.

User question:
{question}

Current evidence summary (may be empty):
{evidence_summary}

Steps taken: {step}/{max_steps}

Output format (STRICT):
Thought: <one short sentence about what's missing or why evidence is sufficient>
Action: retrieve[your query here]
or
Action: FINAL
"""


# ============================================================
# Data Classes
# ============================================================

@dataclass
class AgentState:
    route: str = ""
    docs_found: int = 0
    citations: List[str] = field(default_factory=list)
    evidence_score: int = 0
    verification_score: int = 0
    decomposed: bool = False
    sub_questions: List[str] = field(default_factory=list)


# ============================================================
# Agent Class
# ============================================================

class StudyCopilotAgent:
    """
    StudyCopilot Agent with optimized architecture:
    1. Post-Retrieval Routing
    2. Progressive 3-layer Gate
    3. Query Decomposition
    4. Self-Verification
    """

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.llm = OllamaLLM(
            model=cfg.llm_model,
            temperature=0.2,
            base_url="http://127.0.0.1:11434"
        )

        # 两个检索器
        self.quick_retrieve = build_quick_retriever(cfg)
        self.strong_retrieve = build_strong_retriever(cfg)

        # 配置参数
        self.react_max_steps = getattr(cfg, "react_max_steps", 3)
        self.enable_decomposition = getattr(cfg, "enable_decomposition", True)
        self.enable_verification = getattr(cfg, "enable_verification", True)
        self.max_sub_questions = getattr(cfg, "max_sub_questions", 3)
        self.evidence_threshold = getattr(cfg, "evidence_score_threshold", 40)
        self.verification_threshold = getattr(cfg, "verification_score_threshold", 70)
        self.max_refine_attempts = getattr(cfg, "max_refine_attempts", 1)

    # ============================================================
    # Utility Methods
    # ============================================================

    def _is_chinese(self, text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    @staticmethod
    def _doc_key(d: Document) -> Tuple[Any, Any, str]:
        """Document fingerprint, used to merge/dedup docs across retrieval rounds."""
        return (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:100])

    def _make_evidence_preview(self, docs: List[Document], max_chars: int = 300) -> str:
        """生成证据预览，用于路由决策"""
        if not docs:
            return "(No documents retrieved)"

        previews = []
        for i, d in enumerate(docs[:3], start=1):
            snippet = (d.page_content or "").strip().replace("\n", " ")[:max_chars]
            previews.append(f"[{i}] {snippet}...")
        return "\n".join(previews)

    def _make_evidence_summary(self, docs: List[Document], max_chars_per_doc: int = 180) -> str:
        """生成证据摘要，用于 ReAct planner"""
        if not docs:
            return "(empty)"

        lines = []
        for i, d in enumerate(docs[:4], start=1):
            snippet = (d.page_content or "").strip().replace("\n", " ")
            if len(snippet) > max_chars_per_doc:
                snippet = snippet[:max_chars_per_doc] + "..."
            src = d.metadata.get("source", "")
            page = d.metadata.get("page", None)
            ref = f"{src} p{page}" if page is not None else src
            lines.append(f"[{i}] {ref} :: {snippet}")
        return "\n".join(lines)

    # ============================================================
    # Gate 1: Quick Retrieve + Post-Retrieval Routing
    # ============================================================

    def _post_retrieval_route(self, question: str, preview_docs: List[Document]) -> str:
        """基于预检索证据决定路由"""
        evidence_preview = self._make_evidence_preview(preview_docs)

        prompt = POST_RETRIEVAL_ROUTER_PROMPT.format(
            question=question,
            evidence_preview=evidence_preview
        )

        resp = self.llm.invoke(prompt).strip().upper()

        # 只保留两条有真实行为差异的路由；默认 RAG
        # 注意：NO_RAG 必须先判，否则 "NO_RAG" 里的 "RAG" 会先命中
        if "NO_RAG" in resp:
            return "NO_RAG"
        return "RAG"

    # ============================================================
    # Query Decomposition
    # ============================================================

    def _decompose_question(self, question: str) -> Tuple[bool, List[str]]:
        """分解复杂问题为子问题"""
        if not self.enable_decomposition:
            return False, [question]

        prompt = DECOMPOSE_PROMPT.format(
            question=question,
            max_sub=self.max_sub_questions
        )

        try:
            resp = self.llm.invoke(prompt).strip()
            # 尝试解析 JSON
            data = json.loads(resp)
            needs_decomp = data.get("needs_decomposition", False)
            sub_qs = data.get("sub_questions", [question])

            if needs_decomp and len(sub_qs) > 1:
                return True, sub_qs[:self.max_sub_questions]
        except (json.JSONDecodeError, KeyError):
            pass

        return False, [question]

    # ============================================================
    # Gate 2: Evidence Scoring
    # ============================================================

    def _score_evidence(self, question: str, context: str) -> int:
        """评估证据质量分数 (0-100)"""
        # 硬阈值检查
        min_chars = getattr(self.cfg, "min_evidence_chars", 100)
        if len(context) < min_chars:
            return 0

        prompt = EVIDENCE_SCORE_PROMPT.format(
            question=question,
            context=context
        )

        try:
            resp = self.llm.invoke(prompt).strip()

            # 取「最后一个」数字而不是第一个：模型常写成
            #   "On a 0-100 scale, I'd say 75"  → 取第一个会得到 0，直接误判证据不足
            # 先剥掉量表说明（0-100 / out of 100），再取最后一个落在 0-100 的数字
            cleaned = re.sub(r"\b0\s*(?:-|to|–)\s*100\b", " ", resp, flags=re.IGNORECASE)
            cleaned = re.sub(r"\bout\s+of\s+100\b", " ", cleaned, flags=re.IGNORECASE)

            nums = [int(n) for n in re.findall(r"\d+", cleaned)]
            nums = [n for n in nums if 0 <= n <= 100]
            if nums:
                return nums[-1]
        except Exception:
            pass

        return 50  # 默认中等分数

    # ============================================================
    # Gate 3: Self-Verification
    # ============================================================

    def _verify_answer(self, question: str, answer: str, context: str) -> Tuple[int, List[str]]:
        """验证答案质量，返回 (score, issues)"""
        if not self.enable_verification:
            return 100, []

        prompt = VERIFY_PROMPT.format(
            context=context,
            question=question,
            answer=answer
        )

        try:
            resp = self.llm.invoke(prompt).strip()
            data = json.loads(resp)
            score = data.get("overall_score", 70)
            issues = data.get("issues", [])
            return score, issues
        except (json.JSONDecodeError, KeyError):
            return 70, []

    def _refine_answer(self, question: str, previous_answer: str,
                       context: str, issues: List[str]) -> str:
        """修复答案中的问题"""
        prompt = REFINE_PROMPT.format(
            context=context,
            question=question,
            previous_answer=previous_answer,
            issues="\n".join(f"- {issue}" for issue in issues)
        )

        resp = self.llm.invoke(prompt)
        return (resp or "").strip()

    # ============================================================
    # ReAct Retrieve Loop
    # ============================================================

    def _parse_planner_action(self, text: str) -> Tuple[str, Optional[str]]:
        """解析 planner 输出"""
        t = (text or "").strip()

        if re.search(r"Action:\s*FINAL\b", t, flags=re.IGNORECASE):
            return ("final", None)

        m = re.search(r"Action:\s*retrieve\[(.+?)\]\s*$", t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            q = m.group(1).strip()
            if q:
                return ("retrieve", q)

        return ("unknown", None)

    def _react_retrieve_loop(self, question: str, initial_docs: List[Document] = None
                             ) -> Tuple[List[Document], List[str], List[str]]:
        """
        ReAct 检索循环
        改进：首轮强制检索，避免跳过
        """
        # 用 list(...) 拷贝：best_docs 后面会被就地追加，
        # 直接引用会污染调用方传进来的 preview_docs
        best_docs: List[Document] = list(initial_docs) if initial_docs else []
        all_queries: List[str] = []
        debug_notes: List[str] = []
        seen_queries: set = set()

        # 已收录文档的指纹，用于跨轮合并去重
        seen_doc_keys = {self._doc_key(d) for d in best_docs}

        def merge_docs(new_docs: List[Document]) -> int:
            """把新检索到的文档并入 best_docs（去重），返回新增条数"""
            added = 0
            for d in new_docs:
                key = self._doc_key(d)
                if key not in seen_doc_keys:
                    seen_doc_keys.add(key)
                    best_docs.append(d)
                    added += 1
            return added

        # 如果没有初始文档，强制首轮检索
        if not best_docs:
            debug_notes.append("[react] No initial docs, forcing first retrieval")
            docs, queries_used, dbg = self.strong_retrieve(question)
            merge_docs(docs)
            all_queries.extend(queries_used or [question])
            debug_notes.extend(dbg or [])
            seen_queries.add(question.lower().strip())

        evidence_summary = self._make_evidence_summary(best_docs)

        # ReAct 循环
        for step in range(1, self.react_max_steps + 1):
            planner_prompt = REACT_PLANNER_PROMPT.format(
                question=question,
                evidence_summary=evidence_summary,
                step=step,
                max_steps=self.react_max_steps
            )

            planner_out = self.llm.invoke(planner_prompt).strip()
            action, payload = self._parse_planner_action(planner_out)

            debug_notes.append(f"[planner step {step}] {planner_out[:200]}")

            if action == "final":
                debug_notes.append(f"[planner step {step}] Decided FINAL, stopping retrieval")
                break

            if action != "retrieve" or not payload:
                payload = question
                debug_notes.append(f"[planner step {step}] Malformed action, using original question")

            # 重复 query 检测
            norm_q = payload.lower().strip()
            if norm_q in seen_queries:
                debug_notes.append(f"[planner step {step}] Repeated query, stopping")
                break

            seen_queries.add(norm_q)

            # 执行检索
            docs, queries_used, dbg = self.strong_retrieve(payload)
            all_queries.extend(queries_used or [payload])
            debug_notes.extend(dbg or [])

            # 合并文档：去重累加，而不是「谁多用谁」整批替换
            # （旧写法下，本轮检索到的新证据只要条数不占优就会被整批丢弃，
            #   多步检索等于白跑）
            added = merge_docs(docs)
            debug_notes.append(
                f"[planner step {step}] merged {added} new docs (total={len(best_docs)})"
            )

            evidence_summary = self._make_evidence_summary(best_docs)

        return best_docs, all_queries, debug_notes

    # ============================================================
    # Fallback Answer
    # ============================================================

    def _fallback_answer(self, question: str, partial_context: str = "") -> str:
        """生成 fallback 回答"""
        if partial_context:
            prompt = f"""You are a helpful assistant.
            
The user asked a question, but the retrieved evidence may not be sufficient.
You may use the partial evidence below if helpful, but be cautious.

Partial Evidence:
{partial_context[:500]}

Rules:
- Detect the language of the QUESTION and answer in that language.
- If evidence is helpful, use it. Otherwise, use general knowledge.
- Be honest about uncertainty.
- Do NOT invent citations.

QUESTION:
{question}

Answer:
"""
        else:
            prompt = FALLBACK_PROMPT.format(question=question)

        resp = self.llm.invoke(prompt)
        return (resp or "").strip()

    # ============================================================
    # Main Answer Pipeline
    # ============================================================

    def answer(self, question: str) -> Dict[str, Any]:
        """
        主回答流程（优化版架构）

        流程：
        1. Query Analysis (Decomposition)
        2. Gate 1: Quick Retrieve + Post-Retrieval Routing
        3. Deep Retrieve (ReAct Loop)
        4. Gate 2: Evidence Scoring
        5. Answer Generation
        6. Gate 3: Self-Verification
        """
        debug_notes: List[str] = []
        state = AgentState()

        # ========================================
        # Step 1: Query Decomposition
        # ========================================
        decomposed, sub_questions = self._decompose_question(question)
        state.decomposed = decomposed
        state.sub_questions = sub_questions
        debug_notes.append(f"[decompose] decomposed={decomposed}, sub_questions={sub_questions}")

        # ========================================
        # Step 2: Gate 1 - Quick Retrieve + Routing
        # ========================================
        preview_docs = self.quick_retrieve(question)
        debug_notes.append(f"[gate1] quick_retrieve found {len(preview_docs)} docs")

        # 如果预检索完全没有结果，早期退出
        if not preview_docs:
            debug_notes.append("[gate1] No docs found, early exit to fallback")
            fb = self._fallback_answer(question)
            msg = "**在你的笔记里没有找到相关证据。**" if self._is_chinese(question) else \
                "**No relevant evidence found in your notes.**"
            return {
                "final": msg + "\n\n" + fb,
                "route": "EARLY_EXIT",
                "citations": [],
                "context": "",
                "docs_found": 0,
                "fallback": True,
                "debug": debug_notes,
                "queries_used": [],
                "evidence_score": 0,
                "verification_score": None,
            }

        # Post-Retrieval Routing
        route = self._post_retrieval_route(question, preview_docs)
        state.route = route
        debug_notes.append(f"[gate1] post_retrieval_route = {route}")

        # ========================================
        # Route: NO_RAG - 直接用通用知识回答
        # ========================================
        if route == "NO_RAG":
            final = self.llm.invoke(
                "Answer briefly and safely. If unsure, say you are unsure.\n\nQuestion:\n" + question
            ).strip()
            final = "⚠️ **Answered without note evidence (NO_RAG).**\n\n" + final
            return {
                "final": final,
                "route": route,
                "citations": [],
                "context": "",
                "docs_found": 0,
                "fallback": True,
                "debug": debug_notes,
                "queries_used": [],
                "evidence_score": 0,
                "verification_score": None,
            }

        # ========================================
        # Step 3: Deep Retrieve (ReAct Loop)
        # ========================================
        # 对于复杂问题，分别检索每个子问题
        all_docs: List[Document] = []
        all_queries: List[str] = []

        questions_to_retrieve = sub_questions if decomposed else [question]

        for q in questions_to_retrieve:
            docs, queries, dbg = self._react_retrieve_loop(q, preview_docs if q == question else None)
            all_docs.extend(docs)
            all_queries.extend(queries)
            debug_notes.extend(dbg)

        # 去重文档
        seen = set()
        unique_docs = []
        for d in all_docs:
            key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:100])
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        docs = unique_docs[:self.cfg.top_k * 2]  # 保留更多文档
        context, citations = format_docs(docs)
        state.docs_found = len(docs)
        state.citations = citations

        debug_notes.append(f"[retrieve] total unique docs = {len(docs)}")

        # ========================================
        # Step 4: Gate 2 - Evidence Scoring
        # ========================================
        if not context:
            debug_notes.append("[gate2] No context, fallback")
            fb = self._fallback_answer(question)
            msg = "**在你的笔记里没有找到相关证据。**" if self._is_chinese(question) else \
                "**No relevant evidence found in your notes.**"
            return {
                "final": msg + "\n\n" + fb,
                "route": route,
                "citations": [],
                "context": "",
                "docs_found": 0,
                "fallback": True,
                "debug": debug_notes,
                "queries_used": all_queries,
                "evidence_score": 0,
                "verification_score": None,
            }

        evidence_score = self._score_evidence(question, context)
        state.evidence_score = evidence_score
        debug_notes.append(f"[gate2] evidence_score = {evidence_score}")

        if evidence_score < self.evidence_threshold:
            debug_notes.append(f"[gate2] Score {evidence_score} < threshold {self.evidence_threshold}, fallback")
            fb = self._fallback_answer(question, context)
            msg = "**证据不足：我在你的笔记里找不到足够内容来可靠回答。**" if self._is_chinese(question) else \
                "**Not enough evidence in your notes to answer confidently.**"

            return {
                "final": msg + "\n\n" + fb,
                "route": route,
                "citations": citations,
                "context": context,
                "docs_found": len(docs),
                "fallback": True,
                "debug": debug_notes,
                "queries_used": all_queries,
                "evidence_score": evidence_score,
                "verification_score": None,
            }

        # ========================================
        # Step 5: Answer Generation
        # ========================================
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        answer = self.llm.invoke(prompt)
        answer = (answer or "").strip()

        debug_notes.append(f"[generate] Answer generated, length={len(answer)}")

        # ========================================
        # Step 6: Gate 3 - Self-Verification
        # ========================================
        # 关闭时返回 None（而不是 100），避免评估报告里出现「满分」假象
        if not self.enable_verification:
            verification_score = None
            debug_notes.append("[gate3] verification disabled (config.enable_verification=False)")
        else:
            verification_score, issues = self._verify_answer(question, answer, context)
            state.verification_score = verification_score
            debug_notes.append(f"[gate3] verification_score = {verification_score}, issues = {issues}")

            # 如果验证分数低，尝试修复
            if verification_score < self.verification_threshold and self.max_refine_attempts > 0:
                debug_notes.append(f"[gate3] Score {verification_score} < threshold, attempting refinement")

                for attempt in range(self.max_refine_attempts):
                    refined_answer = self._refine_answer(question, answer, context, issues)
                    new_score, new_issues = self._verify_answer(question, refined_answer, context)

                    debug_notes.append(f"[gate3] Refinement attempt {attempt + 1}: score {new_score}")

                    if new_score > verification_score:
                        answer = refined_answer
                        verification_score = new_score
                        issues = new_issues

                    if verification_score >= self.verification_threshold:
                        break

        # ========================================
        # Return Final Result
        # ========================================
        return {
            "final": answer,
            "route": route,
            "citations": citations,
            "context": context,          # 供评测审计：答案到底基于什么证据生成
            "docs_found": len(docs),
            "fallback": False,
            "debug": debug_notes,
            "queries_used": all_queries,
            "evidence_score": evidence_score,
            "verification_score": verification_score,
            "decomposed": decomposed,
            "sub_questions": sub_questions if decomposed else [],
        }
