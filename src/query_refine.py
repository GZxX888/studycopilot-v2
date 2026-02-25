# src/query_refine.py
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_query_refine_chain(llm, n_candidates=4):
    """
    输入：用户原始问题 + rewritten query
    输出：选中的 best_query + candidates（json）
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You generate multiple alternative retrieval queries and choose the best one.\n"
         "Goal: maximize recall of relevant lecture-note chunks.\n"
         "Rules:\n"
         f"- Create {n_candidates} candidate queries\n"
         "- Each candidate <= 18 words\n"
         "- Each candidate should use different angle (definition, formula, example, related terms)\n"
         "- Choose ONE best_query among them\n"
         "- Do NOT answer the user\n"
         "- Output STRICT JSON only with keys: candidates, best_query\n"
         'Example: {"candidates":["..."],"best_query":"..."}'),
        ("human",
         "User question:\n{question}\n\n"
         "Rewritten query:\n{rewritten_query}")
    ])

    chain = prompt | llm | StrOutputParser()

    def _safe_parse(text: str):
        text = text.strip()
        try:
            data = json.loads(text)
            # minimal validation
            if "best_query" not in data:
                raise ValueError("missing best_query")
            if "candidates" not in data or not isinstance(data["candidates"], list):
                data["candidates"] = []
            return data
        except Exception:
            # fallback: just use rewritten_query if JSON broken
            return {"candidates": [], "best_query": ""}

    return chain, _safe_parse