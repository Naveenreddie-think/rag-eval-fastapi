"""
Streamlit Web Application: Interactive FastAPI Docs RAG & Evaluation Engine.

Provides an interactive user interface for:
1. Live Grounded Q&A with live retrieval inspection and latency profiling.
2. Evaluation Explorer with interactive metric cards, charts, and ablation results.
3. Security & Poisoned Document test harness viewer.

Run with:
    python -m streamlit run streamlit_app.py
"""

import json
import time
from pathlib import Path
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="FastAPI Docs RAG & Evaluation Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
    }
    .citation-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-primary { background-color: #1f6feb; color: #ffffff; }
    .badge-success { background-color: #238636; color: #ffffff; }
    .badge-warning { background-color: #d29922; color: #0d1117; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helper Functions ---

@st.cache_resource
def load_bm25():
    from app.retrieval.bm25 import build_bm25_index
    chunks_path = Path("data/processed/chunks.json")
    if chunks_path.exists():
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        build_bm25_index(chunks)
        return len(chunks)
    return 0

chunks_count = load_bm25()

# --- Sidebar Controls ---
st.sidebar.title("⚡ RAG Pipeline Controls")
mode = st.sidebar.selectbox(
    "Retrieval Mode",
    ["Hybrid (Blended α=0.7)", "Dense-Only (BGE-M3)", "BM25-Only (Okapi)", "Hybrid (Pure Rerank α=1.0)"],
    index=0,
)

top_k = st.sidebar.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=5)

blend_alpha = 0.7
if "α=1.0" in mode:
    blend_alpha = 1.0
elif "α=0.7" in mode:
    blend_alpha = 0.7

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Indexed Chunks:** `{chunks_count}`")
st.sidebar.markdown("**Embeddings:** `BAAI/bge-m3` (1024d)")
st.sidebar.markdown("**Reranker:** `ms-marco-MiniLM-L-6-v2`")
st.sidebar.markdown("**Generator:** `Qwen2.5-3B-Instruct` (bf16)")

# --- Main App Layout ---
st.title("⚡ FastAPI Docs RAG & Evaluation Engine")
st.caption("Empirically evaluated, structure-aware RAG pipeline with grounded answering and real-time latency diagnostics.")

tab1, tab2, tab3 = st.tabs(["💬 Interactive Q&A Assistant", "📊 Benchmark & Evaluation Results", "🛡️ Security Evaluation"])

# ================= TAB 1: INTERACTIVE ASSISTANT =================
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input(
            "Ask a technical question about FastAPI:",
            placeholder="e.g., How do you declare an integer path parameter in FastAPI?",
        )
    with col2:
        st.write("")
        st.write("")
        ask_btn = st.button("🚀 Ask Assistant", use_container_width=True)

    # Sample starter buttons
    st.markdown("**Quick Queries:**")
    q_cols = st.columns(3)
    sample_queries = [
        "How do path parameters work with type annotations?",
        "How do you schedule background tasks to run after response?",
        "How do you connect FastAPI directly to Apache Kafka?",  # no_answer query
    ]
    
    for idx, q_text in enumerate(sample_queries):
        if q_cols[idx].button(q_text, key=f"quick_q_{idx}"):
            user_query = q_text
            ask_btn = True

    if ask_btn and user_query:
        with st.spinner("Retrieving context and generating grounded answer..."):
            from app.generation.generator import generate_answer
            from app.retrieval.bm25 import bm25_search
            from app.retrieval.dense import dense_search
            from app.retrieval.hybrid import hybrid_search
            from app.eval.faithfulness import is_refusal_response

            latencies = {}
            t0 = time.perf_counter()

            if "Dense" in mode:
                retrieved = dense_search(user_query, top_k=top_k)
                latencies["Dense Search"] = (time.perf_counter() - t0) * 1000
            elif "BM25" in mode:
                retrieved = bm25_search(user_query, top_k=top_k)
                latencies["BM25 Search"] = (time.perf_counter() - t0) * 1000
            else:
                out = hybrid_search(user_query, top_k=top_k, blend_alpha=blend_alpha)
                retrieved = out["results"]
                latencies["Dense Search"] = out["latency_ms"].get("dense_search_ms", 0)
                latencies["BM25 Search"] = out["latency_ms"].get("bm25_search_ms", 0)
                latencies["RRF Fusion"] = out["latency_ms"].get("fusion_ms", 0)
                latencies["Cross-Encoder Rerank"] = out["latency_ms"].get("rerank_ms", 0)

            latencies["Total Retrieval"] = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            gen = generate_answer(user_query, retrieved)
            answer = gen["answer"]
            latencies["Generation"] = (time.perf_counter() - t1) * 1000
            latencies["Total End-to-End"] = (time.perf_counter() - t0) * 1000

            refusal = is_refusal_response(answer)

        # Output Layout
        st.markdown("---")
        st.subheader("💡 Answer")
        if refusal:
            st.warning("⚠️ **Honest Refusal**: The model determined that the retrieved context does not contain sufficient documentation to answer this question.")
        
        st.markdown(answer)

        # Latency Metrics Dashboard
        st.markdown("### ⏱️ Latency Profile")
        m_cols = st.columns(len(latencies))
        for idx, (stage, ms) in enumerate(latencies.items()):
            m_cols[idx].metric(stage, f"{ms:.1f} ms")

        # Retrieved Context Citations
        st.markdown("### 📚 Retrieved Citations & Context")
        for rank, chunk in enumerate(retrieved, start=1):
            with st.expander(f"#{rank} • {chunk['source_path']} › {chunk['header_path']}"):
                score_info = []
                if "final_score" in chunk:
                    score_info.append(f"**Blended Score:** `{chunk['final_score']:.3f}`")
                if "rerank_score" in chunk:
                    score_info.append(f"**Rerank Logit:** `{chunk['rerank_score']:.3f}`")
                if "score" in chunk:
                    score_info.append(f"**Cosine Sim:** `{chunk['score']:.3f}`")
                
                if score_info:
                    st.markdown(" | ".join(score_info))
                
                st.code(chunk["text"], language="markdown")


# ================= TAB 2: BENCHMARK & EVALUATION RESULTS =================
with tab2:
    st.subheader("📈 Ground Truth Evaluation (80-Pair Benchmark)")
    st.caption("Measured results across 50 single-hop, 15 multi-hop, and 15 no-answer questions.")

    retrieval_res_path = Path("data/processed/retrieval_eval_results.json")
    if retrieval_res_path.exists():
        eval_data = json.loads(retrieval_res_path.read_text(encoding="utf-8"))

        summary_rows = []
        for cfg_name, res in eval_data.items():
            o = res["overall"]
            summary_rows.append({
                "Configuration": cfg_name,
                "Hit Rate@5": f"{o['hit_rate']:.3f}",
                "Recall@5": f"{o['recall']:.3f}",
                "Precision@5": f"{o['precision']:.3f}",
                "MRR": f"{o['mrr']:.3f}",
            })
        st.table(summary_rows)
    else:
        st.info("Run `python -m scripts.run_eval` to generate the full ablation benchmark.")

    st.markdown("---")
    st.subheader("🎯 Key Architectural Findings")
    st.markdown(
        """
        - **Dense-Only Retrieval is Strongest for Hit Rate & Recall**: `BGE-M3` dense search achieves higher recall (0.692) than out-of-the-box hybrid search.
        - **Cross-Encoder Header Demotion**: Pre-trained cross-encoders tend to over-index on vocabulary headers; blending RRF scores with $\\alpha=0.7$ restores stability.
        - **Refusal Discipline**: Constrained prompt engineering prevents hallucinations on out-of-domain topics (tested on 15 `no_answer` queries).
        """
    )


# ================= TAB 3: SECURITY EVALUATION =================
with tab3:
    st.subheader("🛡️ Step 6 Security Evaluation: Indirect Prompt Injection")
    st.caption("Testing RAG resilience against poisoned corpus injection attacks.")

    sec_res_path = Path("data/processed/security_eval_results.json")
    if sec_res_path.exists():
        sec_data = json.loads(sec_res_path.read_text(encoding="utf-8"))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Scenarios Tested", sec_data.get("total_scenarios", 0))
        c1.metric("Retrieval Penetration", f"{sec_data.get('retrieval_penetration_rate', 0)*100:.1f}%")
        c3.metric("Attack Success Rate (ASR)", f"{sec_data.get('attack_success_rate', 0)*100:.1f}%")

        st.markdown("#### Scenario Breakdown")
        for sc in sec_data.get("scenarios", []):
            status_badge = "❌ VULNERABLE" if sc["attack_successful"] else "✅ RESISTED"
            with st.expander(f"Scenario [{sc['scenario_id']}] - {status_badge}"):
                st.markdown(f"**Target Query:** `{sc['target_query']}`")
                st.markdown(f"**Penetrated Retrieval:** `{sc['poison_penetrated_retrieval']}`")
                st.markdown(f"**Canary Surfaced:** `{sc['canary_surfaced_in_answer']}`")
                st.markdown(f"**Generated Output:**\n> {sc['generated_answer']}")
    else:
        st.info("Run `python -m scripts.run_security_eval` to execute the security test suite.")
