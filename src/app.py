
from __future__ import annotations

import streamlit as st
from config import RAGConfig
from agent import StudyCopilotAgent


from pathlib import Path
import shutil


from ingest import run_ingest



st.set_page_config(page_title="StudyCopilot-v2", layout="wide")
st.title("StudyCopilot-v2 — Minimal Agent + RAG (Ollama)")
st.caption("Config-driven: uses src/config.py (embedding / vectordb / llm)")

cfg = RAGConfig()

with st.sidebar:
    st.header("Current Config")
    st.markdown("---")
    st.subheader("Knowledge Base")

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs to data/",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        data_dir = Path(RAGConfig().data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for f in uploaded_files:
            save_path = data_dir / f.name
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            saved += 1

        st.sidebar.success(f"Uploaded {saved} PDF(s) to {data_dir}")

    # 一键重建按钮
    if st.button("Rebuild Knowledge Base", type="primary"):
        cfg = RAGConfig()

        # 1) 清空 vectordb（从零重建，最稳）
        if cfg.vectordb_dir.exists():
            shutil.rmtree(cfg.vectordb_dir, ignore_errors=True)
        cfg.vectordb_dir.mkdir(parents=True, exist_ok=True)

        # 2) 跑 ingest
        with st.spinner("Rebuilding VectorDB..."):
            stats = run_ingest(cfg)

        st.sidebar.success("✅ Knowledge base rebuilt!")

        # 3) 重建 agent（刷新 retriever）
        st.session_state.agent = StudyCopilotAgent(cfg)

        # 4) 强制刷新页面
        st.rerun()
    st.write(f"**LLM:** {cfg.llm_model}")
    st.write(f"**Embedding:** {cfg.embedding_model}")
    st.write(f"**vectordb:** {cfg.vectordb_dir}")
    st.write(f"**top_k:** {cfg.top_k}")
    st.divider()
    if st.button("Rebuild Agent"):
        st.session_state.agent = StudyCopilotAgent(cfg)
        st.success("Agent rebuilt.")

if "agent" not in st.session_state:
    st.session_state.agent = StudyCopilotAgent(cfg)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question about your notes...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = None  # ✅ 先定义，避免异常后引用不到

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent Processing: Route → Retrieve → Gate → Generate → Verify..."):
            try:
                result = st.session_state.agent.answer(prompt)
                st.markdown(result["final"])
                # ✅ 成功后立刻写入历史
                st.session_state.messages.append({"role": "assistant", "content": result["final"]})
            except Exception as e:
                err = f"❌ Agent error: {type(e).__name__}: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.stop()  # ✅ 不要 raise，让应用继续活着

        if result is not None:
            with st.expander("🔍 Debug Info"):
                # 基本信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Route", result.get('route', 'N/A'))
                with col2:
                    st.metric("Docs Found", result.get('docs_found', 0))
                with col3:
                    fallback_status = "Yes" if result.get('fallback', True) else "No"
                    st.metric("Fallback", fallback_status)

                # 分数指标（Gate 2 证据打分）
                # 自验证已在 config 里关闭，verification_score 为 None，故不展示
                st.divider()
                st.subheader("📊 Quality Scores")
                evidence_score = result.get("evidence_score") or 0
                st.progress(evidence_score / 100, text=f"Evidence Score: {evidence_score}/100")

                # queries_used
                if result.get("queries_used"):
                    st.divider()
                    st.subheader("🔎 Queries Used")
                    for q in result["queries_used"][:10]:  # 限制显示数量
                        st.write(f"- {q}")
                    if len(result["queries_used"]) > 10:
                        st.write(f"... and {len(result['queries_used']) - 10} more")

                # citations
                if result.get("citations"):
                    st.divider()
                    st.subheader("📚 Citations")
                    for c in result["citations"]:
                        st.write(c)

                # planner/debug steps
                if result.get("debug"):
                    st.divider()
                    st.subheader("🛠️ Debug Log")
                    with st.expander("Show detailed debug log"):
                        for line in result["debug"]:
                            st.text(line)