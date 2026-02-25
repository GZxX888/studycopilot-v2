# src/query_rewrite.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_query_rewrite_chain(llm):
    """
    输入：用户原始问题
    输出：更适合向量检索的 rewritten query（更关键词化、去口语、保留核心名词）
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You rewrite user questions into a concise search query for retrieving lecture notes.\n"
         "Rules:\n"
         "- Keep it short (<= 25 words)\n"
         "- Prefer technical keywords and nouns\n"
         "- Remove filler words\n"
         "- If the question is multi-part, keep only the part most likely answered by notes\n"
         "- Do NOT answer the question\n"
         "Return ONLY the rewritten query."),
        ("human", "{question}")
    ])

    return prompt | llm | StrOutputParser()