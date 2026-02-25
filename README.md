# 📘 StudyCopilot-v2  
### Agent-Enhanced RAG System (Ollama + Llama3)

---

# 🧠 Project Overview | 项目简介

## 🇬🇧 English

StudyCopilot-v2 is an advanced Retrieval-Augmented Generation (RAG) system powered by a local LLM (Ollama + Llama3).  

Unlike a basic RAG pipeline, this version integrates:

- ReAct Agent framework  
- Query Refinement  
- Hybrid Retrieval  
- Evidence Gate (anti-hallucination control)  
- Multilingual question support  

The system is designed for course-based knowledge QA, emphasizing faithfulness, interpretability, and reliability.

---

## 🇨🇳 中文

StudyCopilot-v2 是一个基于本地大模型（Ollama + Llama3）的增强型 RAG 系统。

相比基础 RAG，本版本引入了：

- ReAct Agent 框架  
- Query Refinement 查询改写机制  
- Hybrid 混合检索  
- Evidence Gate 证据门控机制（防幻觉）  
- 多语言问答支持  

该系统专为课程知识问答场景设计，强调：

- 忠实于课程资料  
- 可解释性  
- 幻觉控制  
- 产品级稳定性  

---

# 🚀 Core Capabilities | 核心能力

---

## 1️⃣ ReAct Agent Framework

### English

The system uses the ReAct (Reasoning + Acting) paradigm:

Thought → Action → Observation loop

Capabilities:

- Multi-step retrieval  
- Dynamic query reformulation  
- Evidence-aware reasoning  
- Controlled output routing (RAG / fallback)

This enables structured reasoning rather than single-pass generation.

---

### 中文

系统采用 ReAct（Reasoning + Acting）框架：

思考 → 行动 → 观察 循环机制

能力包括：

- 多步检索  
- 动态 query 改写  
- 基于证据强度决策  
- 控制输出路径（RAG 或 fallback）

这使系统具备结构化推理能力，而不是单次生成。

---

## 2️⃣ Query Refinement

### English

Query refinement improves retrieval quality through:

- Keyword extraction  
- Multi-query expansion  
- Cross-lingual enhancement  
- Best-query selection  

This improves recall stability, especially for multilingual queries.

---

### 中文

查询改写机制包括：

- 关键词提取  
- 多 query 扩展  
- 中英检索增强  
- 最优 query 选择  

显著提升跨语言检索稳定性。

---

## 3️⃣ Hybrid Retrieval

### English

The system combines:

- Dense vector retrieval (Sentence-Transformers)  
- BM25 keyword retrieval  
- FlashRank reranking  

Advantages:

- Semantic + lexical matching  
- Higher recall robustness  
- Reduced missing relevant chunks  

---

### 中文

系统采用混合检索：

- 向量语义检索  
- BM25 关键词检索  
- FlashRank 重排序  

优势：

- 语义 + 关键词双通道  
- 提升召回覆盖率  
- 降低漏召回风险  

---

## 4️⃣ Evidence Gate (Anti-Hallucination)

### English

An evidence validation layer checks:

- Retrieved context length  
- Evidence sufficiency  
- Citation validity  

If evidence is insufficient, the system blocks confident generation.

---

### 中文

证据门控机制会：

- 统计召回文本长度  
- 判断证据是否充足  
- 控制是否允许回答  

证据不足时拒绝自信回答，从而降低幻觉风险。

---

## 5️⃣ Multilingual Support

### English

Supports:

- English questions  
- Chinese questions  
- Automatic language-aligned answers  
- Cross-lingual retrieval enhancement  

---

### 中文

支持：

- 英文提问  
- 中文提问  
- 自动语言对齐回答  
- 跨语言检索增强  

---

# 🏗 System Architecture | 系统架构

```
User Question
      ↓
ReAct Agent Planner
      ↓
Query Refinement
      ↓
Hybrid Retrieval
      ↓
Reranking
      ↓
Evidence Gate
      ↓
Answer Generation (LLM)
```

---

# ⚙️ Tech Stack | 技术栈

### Backend

- Python  
- LangChain  
- Ollama (Llama3)  
- ChromaDB  
- FlashRank  

### Retrieval

- sentence-transformers/all-MiniLM-L6-v2  
- BM25  
- Hybrid Search  

### Frontend

- Streamlit  

---

# 🛡 Hallucination Control Strategy | 幻觉控制策略

- Strict "Use ONLY the CONTEXT" prompt rule  
- Evidence length threshold  
- Citation enforcement  
- Fallback isolation  
- Structured output formatting  

---

# 📊 Product-Level Features | 产品级能力体现

This system demonstrates:

- Agent orchestration  
- RAG engineering  
- Retrieval optimization  
- Hallucination mitigation  
- Multilingual robustness  
- Modular configuration design  

---

# 🎯 Highlights | 项目亮点

## English

- Local LLM deployment  
- Agent + RAG deep integration  
- Multi-step reasoning  
- Hybrid retrieval optimization  
- Industrial-style hallucination control  
- Cross-language robustness  

## 中文

- 本地大模型部署  
- Agent 与 RAG 深度融合  
- 多步推理机制  
- 混合检索优化  
- 工业级幻觉控制  
- 跨语言稳定性增强  

---

# 🔮 Future Improvements | 未来优化方向

- Automatic translation-based retrieval  
- Confidence scoring for retrieval  
- Self-evaluation loop  
- Adaptive evidence threshold  
- UI-level citation visualization  

---

# 🏁 Conclusion | 总结

StudyCopilot-v2 is not a simple RAG demo.

It is an Agent-driven, multilingual, hybrid-retrieval QA system designed with product-level robustness and hallucination control.

StudyCopilot-v2 不只是一个基础 RAG 示例，而是一个融合 Agent 推理、多语言增强、混合检索与幻觉控制机制的产品级问答系统。

It reflects practical LLM engineering, RAG optimization, and real-world AI product thinking.
