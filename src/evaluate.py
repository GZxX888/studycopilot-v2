# src/evaluate.py
import json
from pathlib import Path
from typing import Dict, Any, List

from config import RAGConfig
from agent import StudyCopilotAgent


def _is_refused_or_fallback(result: Dict[str, Any]) -> bool:
    """用 agent 返回的 fallback 作为拒答判定（和你当时跑分一致）"""
    if result is None:
        return True
    return bool(result.get("fallback", True))


def run_evaluation() -> List[Dict[str, Any]]:
    cfg = RAGConfig()
    agent = StudyCopilotAgent(cfg)

    base_dir = Path(__file__).resolve().parent.parent
    eval_dir = base_dir / "evaluate"
    eval_file = eval_dir / "eval_questions.json"

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    records: List[Dict[str, Any]] = []

    for item in questions:
        qid = item.get("id", "")
        q = item["question"]
        should_answer = bool(item.get("should_answer", True))
        expected_refusal = bool(item.get("expected_refusal", (not should_answer)))

        result = agent.answer(q)
        refused = _is_refused_or_fallback(result)

        records.append(
            {
                "id": qid,
                "question": q,
                "should_answer": should_answer,
                "expected_refusal": expected_refusal,
                "refused": refused,
                "fallback": result.get("fallback"),
                "route": result.get("route"),
                "docs_found": result.get("docs_found"),
                "queries_used": result.get("queries_used", []),
            }
        )

    return records


def main():
    records = run_evaluation()

    total = len(records)
    should_answer_n = sum(1 for r in records if r["should_answer"] is True)
    should_refuse_n = sum(1 for r in records if r["expected_refusal"] is True)
    refused_n = sum(1 for r in records if r["refused"] is True)

    tp = sum(1 for r in records if r["expected_refusal"] is True and r["refused"] is True)
    fn = sum(1 for r in records if r["expected_refusal"] is True and r["refused"] is False)
    fp = sum(1 for r in records if r["expected_refusal"] is False and r["refused"] is True)

    refusal_recall = tp / max((tp + fn), 1)
    refusal_precision = tp / max((tp + fp), 1)
    false_refusal_rate = fp / max(should_answer_n, 1)

    avg_docs = sum((r.get("docs_found") or 0) for r in records) / max(total, 1)
    avg_queries = sum(len(r.get("queries_used") or []) for r in records) / max(total, 1)

    print("\n========== METRICS ==========")
    print(f"Total questions:                {total}")
    print(f"Should answer:                  {should_answer_n}")
    print(f"Should refuse:                  {should_refuse_n}")
    print(f"Actually refused:               {refused_n}")
    print("----- Refusal -----")
    print(f"Refusal Recall (TPR):           {refusal_recall:.3f}")
    print(f"Refusal Precision:              {refusal_precision:.3f}")
    print(f"False Refusal Rate:             {false_refusal_rate:.3f}")
    print("----- Agent behavior -----")
    print(f"Avg docs_found:                 {avg_docs:.2f}")
    print(f"Avg queries_used:               {avg_queries:.2f}")


if __name__ == "__main__":
    main()