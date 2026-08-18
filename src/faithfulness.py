# -*- coding: utf-8 -*-
"""
确定性 faithfulness 检查（第 1 层）

设计原则：
- 不调用任何 LLM。结果可复现，任何人重跑得到同一个数字。

核心判据：
    幻觉 = 问题的核心实体在语料中不存在
           且 答案对该实体做了「断言」而非「否认」

为什么不用整句词覆盖率：实测过，语料词表仅 2180 词（幻灯片文字稀疏），
导致 supporting/relevant/formulas 这类普通词全部算作 OOV，噪声压倒信号；
同时同义改写会让好答案的覆盖率掉到阈值以下，误报严重。
词覆盖率因此降级为诊断信息，不参与幻觉判定。

用法：
    python src/faithfulness.py evaluate/eval_results_XXXX.jsonl
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

# ============================================================
# 分词 / 语言
# ============================================================

STOPWORDS = {
    "the", "a", "an", "of", "is", "are", "was", "were", "to", "and", "or", "in",
    "on", "at", "for", "with", "that", "this", "these", "those", "it", "its",
    "as", "by", "be", "been", "being", "from", "but", "not", "no", "can", "will",
    "would", "should", "could", "may", "might", "must", "do", "does", "did",
    "have", "has", "had", "we", "you", "they", "he", "she", "i", "there", "here",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "if", "then", "than", "so", "such", "into", "about", "also", "more", "most",
    "some", "any", "all", "each", "other", "one", "two", "used", "using", "use",
    # 答案模板里的高频词，不具备实体性
    "answer", "supporting", "relevant", "formulas", "provides", "applicable",
    "context", "question", "details", "direct", "mentioned", "mention",
}

SCAFFOLD_PATTERNS = [
    r"^\s*\**\s*direct answer", r"^\s*\**\s*supporting details",
    r"^\s*\**\s*relevant formulas", r"^\s*\**\s*answer\s*:",
    r"^\s*i'?m studycopilot", r"^\s*here('s| is) (the|a) ",
]

# 句内否定/免责信号
NEGATION_PATTERNS = [
    r"\bnot\b", r"\bno\b", r"\bnone\b", r"\bnever\b", r"\bwithout\b",
    r"n'?t\b", r"\blacks?\b", r"\babsent\b", r"\bunable\b", r"\bcannot\b",
    r"\binsufficient\b", r"\bunfortunately\b",
    r"没有", r"未[提涉包]", r"不[包含涉]", r"无法", r"缺[乏少]", r"证据不足",
]

CJK = r"一-鿿"


def detect_lang(text: str) -> str:
    if not text:
        return "en"
    cjk = len(re.findall(f"[{CJK}]", text))
    letters = len(re.findall(r"[A-Za-z]", text))
    if cjk == 0:
        return "en"
    return "zh" if cjk >= max(letters * 0.15, 5) else "en"


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    words = [w for w in re.findall(r"[a-z][a-z0-9\-]*", text)
             if len(w) > 2 and w not in STOPWORDS]
    chars = re.findall(f"[{CJK}]", text)
    bigrams = ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)]
    return words + bigrams


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。！？\n])\s*", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def is_scaffold(s: str) -> bool:
    low = s.lower()
    return any(re.search(p, low) for p in SCAFFOLD_PATTERNS)


def has_negation(s: str) -> bool:
    low = s.lower()
    return any(re.search(p, low) for p in NEGATION_PATTERNS)


def term_in(term: str, text: str) -> bool:
    """词边界匹配，兼容复数/派生（autoencoder 命中 autoencoders）"""
    if re.search(f"[{CJK}]", term):
        return term in text
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"[a-z]{0,3}(?![a-z0-9])",
                     text.lower()) is not None


# ============================================================
# 语料词表
# ============================================================

def load_corpus_df(db_path: str) -> Tuple[Counter, int]:
    con = sqlite3.connect(db_path)
    docs = [t for (t,) in con.execute(
        "select string_value from embedding_fulltext_search") if t]
    con.close()
    df = Counter()
    for d in docs:
        df.update(set(tokenize(d)))
    return df, len(docs)


# ============================================================
# 检查 1：引用有效性
# ============================================================

def check_citations(answer: str, n_citations: int) -> Dict[str, Any]:
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer or "")}
    valid = set(range(1, n_citations + 1))
    fabricated = sorted(cited - valid)
    return {
        "cited": sorted(cited),
        "fabricated": fabricated,
        "has_fabricated": bool(fabricated),
        "citation_coverage": round(len(cited & valid) / n_citations, 3) if n_citations else 0.0,
    }


# ============================================================
# 检查 2：问题核心实体是否存在于语料
# ============================================================

def find_missing_entities(question: str, df: Counter, n_docs: int,
                          top_k: int = 3, min_len: int = 4) -> Dict[str, Any]:
    """
    取问题中区分度（IDF）最高的若干词。
    语料 df=0 的词 IDF 视为最大 —— 若这类词是问题核心，
    则整个知识库都不可能回答该问题。
    """
    toks = [t for t in dict.fromkeys(tokenize(question))
            if len(t) >= min_len or re.search(f"[{CJK}]", t)]
    if not toks:
        return {"applicable": False, "reason": "no content tokens", "missing": []}

    def idf(t: str) -> float:
        return math.log((n_docs + 1) / (df.get(t, 0) + 0.5))

    ranked = sorted(toks, key=idf, reverse=True)[:top_k]
    missing = [t for t in ranked if df.get(t, 0) == 0]
    return {
        "applicable": True,
        "top_terms": [{"term": t, "df": df.get(t, 0)} for t in ranked],
        "missing": missing,                       # 语料中完全不存在的核心实体
        "key_entity_absent": bool(missing),
    }


# ============================================================
# 检查 3：答案对缺失实体是「断言」还是「否认」
# ============================================================

def entity_assertion(answer: str, missing: List[str]) -> Dict[str, Any]:
    """
    这是幻觉判定的核心。

    对语料中不存在的实体：
      - 答案说「课件里没有提到 X」        -> 否认，正确行为
      - 答案说「X 是一种……，通过……训练」  -> 断言，幻觉

    已知局限：形如「X 不是线性模型」的句子含否定词，会被误判为否认。
    这类句式在本场景很少见，但若出现会低估幻觉率。
    """
    if not missing:
        return {"applicable": False, "reason": "no missing entity", "assertions": []}

    assertions, denials = [], []
    for s in split_sentences(answer):
        if is_scaffold(s):
            continue
        if not any(term_in(t, s) for t in missing):
            continue
        (denials if has_negation(s) else assertions).append(s.strip()[:200])

    return {
        "applicable": True,
        "n_assertions": len(assertions),
        "n_denials": len(denials),
        "assertions": assertions[:3],
        "denials": denials[:2],
        "is_hallucination": len(assertions) > 0,
    }


# ============================================================
# 诊断（不参与判定）：逐句词覆盖率
# ============================================================

def lexical_diagnostic(answer: str, context: str, thresh: float = 0.5) -> Dict[str, Any]:
    ctx_lang = detect_lang(context)
    ctx_tokens = set(tokenize(context))
    if not ctx_tokens:
        return {"applicable": False, "reason": "empty context"}

    scored, skipped = [], 0
    for s in split_sentences(answer):
        if is_scaffold(s):
            continue
        if detect_lang(s) != ctx_lang:      # 逐句语言判定，跨语言句子跳过
            skipped += 1
            continue
        toks = tokenize(s)
        if len(toks) < 4:
            continue
        scored.append(sum(1 for t in toks if t in ctx_tokens) / len(toks))

    if not scored:
        return {"applicable": False, "reason": "no scorable sentences",
                "skipped_cross_lingual": skipped}
    return {
        "applicable": True,
        "n_sentences": len(scored),
        "skipped_cross_lingual": skipped,
        "mean_coverage": round(sum(scored) / len(scored), 3),
        "low_coverage_sentences": sum(1 for c in scored if c < thresh),
    }


# ============================================================
# 单条审计
# ============================================================

def audit_record(rec: Dict[str, Any], df: Counter, n_docs: int) -> Dict[str, Any]:
    answer = rec.get("answer", "") or ""
    context = rec.get("context", "") or ""
    citations = rec.get("citations", []) or []

    ent = find_missing_entities(rec.get("question", ""), df, n_docs)
    assertion = entity_assertion(answer, ent.get("missing", []))

    # 判定顺序很重要：先看有没有断言，再看是不是拒答。
    # 旧版先判拒答就跳过检查，导致「先编造、后免责」的答案被漏掉。
    if assertion.get("is_hallucination"):
        verdict = "hallucinated"
    elif rec.get("fallback"):
        verdict = "refused_hard"
    elif assertion.get("applicable") and assertion["n_denials"] > 0:
        verdict = "refused_soft"
    else:
        verdict = "answered"

    return {
        "id": rec.get("id"),
        "type": rec.get("type"),
        "tier": rec.get("tier", ""),
        "lang": rec.get("lang", "en"),
        "verdict": verdict,
        "evidence_score": rec.get("evidence_score"),
        "citations_check": check_citations(answer, len(citations)),
        "entity_check": ent,
        "assertion_check": assertion,
        "lexical_diagnostic": lexical_diagnostic(answer, context),
    }


def summarize(audits: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(audits)
    verdicts = Counter(a["verdict"] for a in audits)
    fab = [a for a in audits if a["citations_check"]["has_fabricated"]]
    hall = [a for a in audits if a["verdict"] == "hallucinated"]

    oos = [a for a in audits if a["type"] == "out_of_scope"]
    oos_hall = [a for a in oos if a["verdict"] == "hallucinated"]
    oos_declined = [a for a in oos if a["verdict"] in ("refused_hard", "refused_soft")]
    # 核心实体其实在语料里 -> 该题标注存疑，不应计入 out_of_scope 分母
    oos_suspect = [a for a in oos if not a["entity_check"].get("key_entity_absent")]

    valid = [a for a in oos if a["entity_check"].get("key_entity_absent")]
    valid_hall = [a for a in valid if a["verdict"] == "hallucinated"]

    return {
        "total": n,
        "verdicts": dict(verdicts),
        "citation": {
            "fabricated_count": len(fab),
            "fabrication_rate": round(len(fab) / n, 3) if n else 0,
            "fabricated_ids": [a["id"] for a in fab],
        },
        "hallucination": {
            "count": len(hall),
            "ids": [a["id"] for a in hall],
            "rate_all": round(len(hall) / n, 3) if n else 0,
        },
        "out_of_scope": {
            "total": len(oos),
            "label_suspect": [a["id"] for a in oos_suspect],
            "valid_total": len(valid),
            "declined": len(oos_declined),
            "hallucinated": [a["id"] for a in oos_hall],
            "decline_rate_valid": round(1 - len(valid_hall) / len(valid), 3) if valid else None,
            "hallucination_rate_valid": round(len(valid_hall) / len(valid), 3) if valid else None,
        },
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db = os.path.join(root, "vectordb", "chroma.sqlite3")

    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if "answer" not in recs[0]:
        print("❌ 结果文件未保存 answer/context，无法审计，需用新版 evaluate.py 重跑。")
        sys.exit(1)

    df, n_docs = load_corpus_df(db)
    audits = [audit_record(r, df, n_docs) for r in recs]
    s = summarize(audits)

    print("=" * 78)
    print("确定性 FAITHFULNESS 审计（无 LLM 参与）")
    print("=" * 78)
    print(f"\n语料 {n_docs} chunks / 词表 {len(df)}    样本 {s['total']} 条")
    print(f"行为分类: {s['verdicts']}")

    c = s["citation"]
    print(f"\n--- 检查 1：引用有效性 ---")
    print(f"  编造引用编号: {c['fabricated_count']} 条 ({c['fabrication_rate']*100:.1f}%)  {c['fabricated_ids']}")

    o = s["out_of_scope"]
    print(f"\n--- 检查 2+3：out_of_scope 幻觉判定 ---")
    print(f"  题目总数         : {o['total']}")
    if o["label_suspect"]:
        print(f"  ⚠️ 标注存疑(核心实体其实在语料里): {o['label_suspect']}  -> 已剔除")
    print(f"  有效题数         : {o['valid_total']}")
    print(f"  发生幻觉         : {o['hallucinated']}")
    if o["hallucination_rate_valid"] is not None:
        print(f"  >>> 幻觉率       : {o['hallucination_rate_valid']*100:.1f}%")
        print(f"  >>> 正确拒答率   : {o['decline_rate_valid']*100:.1f}%")

    print("\n" + "=" * 78)
    print("out_of_scope 逐条")
    print("=" * 78)
    print(f"{'id':<6}{'档':<8}{'判定':<15}{'evid':<6}{'缺失实体'}")
    print("-" * 78)
    for a in audits:
        if a["type"] != "out_of_scope":
            continue
        miss = ",".join(a["entity_check"].get("missing", [])) or "(无 → 标注存疑)"
        print(f"{a['id']:<6}{a['tier']:<8}{a['verdict']:<15}{str(a['evidence_score']):<6}{miss}")

    if s["hallucination"]["count"]:
        print("\n" + "=" * 78)
        print("幻觉证据（答案对语料中不存在的实体做了断言）")
        print("=" * 78)
        for a in audits:
            if a["verdict"] != "hallucinated":
                continue
            print(f"\n【{a['id']}】缺失实体: {a['entity_check']['missing']}")
            for s_ in a["assertion_check"]["assertions"]:
                print(f"   ❌ 断言: {s_[:150]}")
            for s_ in a["assertion_check"]["denials"]:
                print(f"   ○ 否认: {s_[:120]}")

    out = path.replace(".jsonl", "_faithfulness.json")
    json.dump({"summary": s, "audits": audits},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n✅ 明细已保存: {os.path.basename(out)}")


if __name__ == "__main__":
    main()
