# src/evaluate.py
"""
StudyCopilot 评估脚本 - 优化版

评估维度：
1. 拒答能力：Refusal Recall, Precision, False Refusal Rate
2. 检索质量：Docs Found, Queries Used, Retrieval Success Rate
3. 答案质量：Evidence Score
4. 路由分布：RAG, NO_RAG, EARLY_EXIT
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

from config import RAGConfig
from agent import StudyCopilotAgent


def _is_refused_or_fallback(result: Dict[str, Any]) -> bool:
    """用 agent 返回的 fallback 作为拒答判定"""
    if result is None:
        return True
    return bool(result.get("fallback", True))


def run_evaluation() -> List[Dict[str, Any]]:
    """运行评估，返回详细记录"""
    cfg = RAGConfig()
    agent = StudyCopilotAgent(cfg)

    base_dir = Path(__file__).resolve().parent.parent
    eval_dir = base_dir / "evaluate"
    eval_file = eval_dir / "eval_questions.json"

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    records: List[Dict[str, Any]] = []

    print(f"\n🚀 Running evaluation on {len(questions)} questions...\n")

    for i, item in enumerate(questions, 1):
        qid = item.get("id", "")
        q = item["question"]
        q_type = item.get("type", "unknown")
        tier = item.get("tier", "")       # out_of_scope 的难度档：easy / medium / hard
        lang = item.get("lang", "en")
        should_answer = bool(item.get("should_answer", True))
        expected_refusal = bool(item.get("expected_refusal", (not should_answer)))

        print(f"  [{i}/{len(questions)}] {qid}: {q[:50]}...")

        result = agent.answer(q)
        refused = _is_refused_or_fallback(result)

        record = {
            "id": qid,
            "question": q,
            "type": q_type,
            "tier": tier,
            "lang": lang,
            "should_answer": should_answer,
            "expected_refusal": expected_refusal,
            "refused": refused,
            "refusal_correct": (refused == expected_refusal),
            "fallback": result.get("fallback"),
            "route": result.get("route"),
            "docs_found": result.get("docs_found", 0),
            "queries_used": result.get("queries_used", []),
            "evidence_score": result.get("evidence_score", 0),
            "verification_score": result.get("verification_score", 0),
            "decomposed": result.get("decomposed", False),
            "sub_questions": result.get("sub_questions", []),
            # ↓ 事后审计用：没有这三项就无法人工复核答案是否有据
            "answer": result.get("final", ""),
            "citations": result.get("citations", []),
            "context": result.get("context", ""),
        }

        records.append(record)

    return records


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算所有评估指标"""
    total = len(records)
    if total == 0:
        return {}

    # === 拒答相关指标 ===
    should_answer_n = sum(1 for r in records if r["should_answer"])
    should_refuse_n = sum(1 for r in records if r["expected_refusal"])
    refused_n = sum(1 for r in records if r["refused"])

    tp = sum(1 for r in records if r["expected_refusal"] and r["refused"])
    fn = sum(1 for r in records if r["expected_refusal"] and not r["refused"])
    fp = sum(1 for r in records if not r["expected_refusal"] and r["refused"])
    tn = sum(1 for r in records if not r["expected_refusal"] and not r["refused"])

    refusal_recall = tp / max(tp + fn, 1)
    refusal_precision = tp / max(tp + fp, 1)
    false_refusal_rate = fp / max(should_answer_n, 1)
    refusal_accuracy = (tp + tn) / total

    # === 检索质量指标 ===
    avg_docs = sum(r.get("docs_found", 0) for r in records) / total
    avg_queries = sum(len(r.get("queries_used", [])) for r in records) / total
    retrieval_success_rate = sum(1 for r in records if r.get("docs_found", 0) > 0) / total

    # === 答案质量指标（仅针对非 fallback 的回答）===
    # 注意：verification_score 在 config.enable_verification=False 时为 None，
    # 不再计入汇总指标（否则会得到一个恒定值，看起来像「满分」）
    answered_records = [r for r in records if not r["refused"]]
    if answered_records:
        avg_evidence_score = sum(r.get("evidence_score", 0) for r in answered_records) / len(answered_records)
    else:
        avg_evidence_score = 0

    # === 路由分布 ===
    route_counter = Counter(r.get("route", "UNKNOWN") for r in records)

    # === 按类型分析 ===
    type_analysis = {}
    type_counter = Counter(r.get("type", "unknown") for r in records)
    for t in type_counter:
        t_records = [r for r in records if r.get("type") == t]
        correct = sum(1 for r in t_records if r["refusal_correct"])
        type_analysis[t] = {
            "total": len(t_records),
            "correct": correct,
            "accuracy": correct / len(t_records) if t_records else 0
        }

    # === out_of_scope 按难度档分析 ===
    # 这是幻觉控制能力的核心口径：库里明确没有的问题，系统拒答了多少
    # 纯行为观测，不依赖任何 LLM 自评
    oos = [r for r in records if r.get("type") == "out_of_scope"]
    tier_analysis = {}
    for t in ("easy", "medium", "hard"):
        t_recs = [r for r in oos if r.get("tier") == t]
        if not t_recs:
            continue
        t_refused = sum(1 for r in t_recs if r["refused"])
        tier_analysis[t] = {
            "total": len(t_recs),
            "refused": t_refused,
            "refusal_rate": t_refused / len(t_recs),
        }

    oos_refused = sum(1 for r in oos if r["refused"])

    # === 跨语言拆分 ===
    lang_analysis = {}
    for lg in sorted({r.get("lang", "en") for r in records}):
        l_recs = [r for r in records if r.get("lang", "en") == lg]
        correct = sum(1 for r in l_recs if r["refusal_correct"])
        lang_analysis[lg] = {
            "total": len(l_recs),
            "correct": correct,
            "accuracy": correct / len(l_recs) if l_recs else 0,
        }

    return {
        "total": total,
        "should_answer": should_answer_n,
        "should_refuse": should_refuse_n,
        "actually_refused": refused_n,
        "out_of_scope": {
            "total": len(oos),
            "refused": oos_refused,
            "refusal_rate": oos_refused / len(oos) if oos else 0.0,
            "by_tier": tier_analysis,
        },
        "by_lang": lang_analysis,
        "refusal": {
            "recall": refusal_recall,
            "precision": refusal_precision,
            "false_refusal_rate": false_refusal_rate,
            "accuracy": refusal_accuracy,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        },
        "retrieval": {
            "avg_docs_found": avg_docs,
            "avg_queries_used": avg_queries,
            "success_rate": retrieval_success_rate,
        },
        "quality": {
            "avg_evidence_score": avg_evidence_score,
            "answered_count": len(answered_records),
        },
        "routing": dict(route_counter),
        "by_type": type_analysis,
    }


def print_metrics(metrics: Dict[str, Any]):
    """打印评估指标"""
    print("\n" + "=" * 60)
    print("📊 EVALUATION METRICS")
    print("=" * 60)

    print("\n----- 基础统计 -----")
    print(f"总问题数:           {metrics['total']}")
    print(f"应该回答:           {metrics['should_answer']}")
    print(f"应该拒答:           {metrics['should_refuse']}")
    print(f"实际拒答:           {metrics['actually_refused']}")

    print("\n----- 拒答能力 (Refusal) -----")
    r = metrics["refusal"]
    print(f"Refusal Recall:     {r['recall']:.3f}  (该拒的拒了多少)")
    print(f"Refusal Precision:  {r['precision']:.3f}  (拒的里面拒对了多少)")
    print(f"False Refusal Rate: {r['false_refusal_rate']:.3f}  (该答的误拒了多少)")
    print(f"Overall Accuracy:   {r['accuracy']:.3f}")

    print("\n----- 混淆矩阵 -----")
    cm = r["confusion_matrix"]
    print(f"              | 实际回答 | 实际拒答 |")
    print(f"  应该回答    |   {cm['tn']:3d}    |   {cm['fp']:3d}    |")
    print(f"  应该拒答    |   {cm['fn']:3d}    |   {cm['tp']:3d}    |")

    print("\n----- 检索质量 (Retrieval) -----")
    ret = metrics["retrieval"]
    print(f"Avg docs_found:     {ret['avg_docs_found']:.2f}")
    print(f"Avg queries_used:   {ret['avg_queries_used']:.2f}")
    print(f"Retrieval Success:  {ret['success_rate']:.3f}  (有检索到文档的比例)")

    print("\n----- 答案质量 (Quality) -----")
    q = metrics["quality"]
    print(f"正确回答数:         {q['answered_count']}")
    print(f"Avg Evidence Score: {q['avg_evidence_score']:.1f}/100")

    print("\n----- 路由分布 (Routing) -----")
    for route, count in sorted(metrics["routing"].items()):
        pct = count / metrics["total"] * 100
        print(f"  {route:15s}: {count:3d} ({pct:.1f}%)")

    print("\n----- 按问题类型分析 -----")
    for t, data in metrics["by_type"].items():
        print(f"  {t:15s}: {data['correct']}/{data['total']} 正确 ({data['accuracy']*100:.1f}%)")

    oos = metrics.get("out_of_scope", {})
    if oos.get("total"):
        print("\n" + "=" * 60)
        print("🎯 幻觉控制能力 (out_of_scope 拒答率)")
        print("=" * 60)
        print(f"  库中确实不存在的问题: {oos['total']} 题")
        print(f"  系统正确拒答:         {oos['refused']} 题")
        print(f"  >>> 拒答率:           {oos['refusal_rate']*100:.1f}%   "
              f"({oos['refused']}/{oos['total']})")
        labels = {"easy": "易 (完全无关领域)",
                  "medium": "中 (同领域未讲过)",
                  "hard": "难 (邻近主题/课务元信息)"}
        print("\n  按难度档:")
        for t, d in oos.get("by_tier", {}).items():
            print(f"    {labels.get(t, t):22s}: {d['refused']}/{d['total']} "
                  f"({d['refusal_rate']*100:5.1f}%)")

    if len(metrics.get("by_lang", {})) > 1:
        print("\n----- 跨语言 -----")
        for lg, d in metrics["by_lang"].items():
            print(f"  {lg:4s}: {d['correct']}/{d['total']} 正确 ({d['accuracy']*100:.1f}%)")


def save_results(records: List[Dict[str, Any]], metrics: Dict[str, Any]):
    """保存评估结果"""
    base_dir = Path(__file__).resolve().parent.parent
    eval_dir = base_dir / "evaluate"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = eval_dir / f"eval_results_{timestamp}.jsonl"

    with open(result_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Results saved to: {result_file}")

    # 也保存 metrics 汇总
    metrics_file = eval_dir / f"eval_metrics_{timestamp}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Metrics saved to: {metrics_file}")


def main():
    records = run_evaluation()
    metrics = compute_metrics(records)
    print_metrics(metrics)
    save_results(records, metrics)


if __name__ == "__main__":
    main()
