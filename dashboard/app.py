"""
LLM Hallucination Guardrail — Streamlit Dashboard

Run with:
    streamlit run dashboard/app.py
"""

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Hallucination Guardrail",
    page_icon="🔍",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔍 LLM Hallucination Guardrail Platform")
st.caption("Real-time hallucination risk scoring — sits between your LLM and users.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API: Online")
        st.write(f"**Model:** {health.get('model_name', 'unknown')}")
        st.write(f"**Loaded:** {'Yes' if health.get('model_loaded') else 'No (heuristic mode)'}")
    except Exception:
        st.error("API: Offline — start with `uvicorn api.main:app`")

    st.divider()
    st.header("Evaluation Metrics")
    try:
        metrics = requests.get(f"{API_URL}/metrics", timeout=3).json()
        for model_name, m in metrics.items():
            with st.expander(model_name):
                col1, col2 = st.columns(2)
                col1.metric("Precision", f"{m['precision_hallucination']:.0%}")
                col2.metric("Recall", f"{m['recall_hallucination']:.0%}")
                col1.metric("AP", f"{m['average_precision']:.2f}")
                col2.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
    except Exception:
        st.info("Run `python train.py` to see metrics.")

    st.divider()
    pr_path = Path(__file__).parent.parent / "reports" / "pr_curve.png"
    if pr_path.exists():
        st.header("PR Curve")
        st.image(str(pr_path))

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_ask, tab_score = st.tabs(["💬 Ask (Live LLM + Guardrail)", "🔎 Score Existing Answer"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ask: question → LLM → guardrail → response
# ══════════════════════════════════════════════════════════════════════════════

with tab_ask:
    st.subheader("Ask a Question — Guardrailed in Real Time")
    st.caption(
        "Your question goes to OpenAI. The answer is scored **before** being shown. "
        "HIGH risk answers are blocked. Requires `OPENAI_API_KEY` in `.env`."
    )

    ask_question = st.text_area(
        "Your question",
        height=80,
        placeholder="e.g. When was the Eiffel Tower built?",
        key="ask_q",
    )
    ask_system = st.text_input(
        "System prompt (optional)",
        placeholder="You are a helpful assistant...",
        key="ask_sys",
    )
    ask_btn = st.button("Ask", type="primary", use_container_width=True, key="ask_btn")

    if ask_btn:
        if not ask_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Calling LLM and running guardrail..."):
                try:
                    payload = {"question": ask_question}
                    if ask_system.strip():
                        payload["system_prompt"] = ask_system
                    resp = requests.post(f"{API_URL}/ask", json=payload, timeout=90)
                    resp.raise_for_status()
                    result = resp.json()
                except requests.HTTPError as e:
                    detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                    st.error(f"API error: {detail}")
                    result = None
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    result = None

            if result:
                risk = result["hallucination_risk"]
                label = result["risk_label"]
                safe = result["safe_to_show"]

                if safe:
                    st.success(f"Risk: {label} ({risk}/100) — Showing answer")
                    st.markdown(f"**Answer:** {result['answer']}")
                elif label == "MEDIUM":
                    st.warning(f"Risk: {label} ({risk}/100) — Showing with caution")
                    st.markdown(f"**Answer:** {result['answer']}")
                    st.info("Verify this answer independently before relying on it.")
                else:
                    st.error(f"Risk: {label} ({risk}/100) — Answer blocked by guardrail")
                    with st.expander("Show blocked answer anyway"):
                        st.markdown(f"**Answer:** {result['answer']}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Risk Score", f"{risk}/100")
                c2.metric("LLM Latency", f"{result['llm_latency_ms']} ms")
                c3.metric("Guardrail Latency", f"{result['guardrail_latency_ms']} ms")
                c4.metric("Total", f"{result['total_latency_ms']} ms")

                feats = result["features"]
                feat_df = pd.DataFrame({
                    "Feature": [
                        "Semantic Similarity",
                        "Confidence Word Density",
                        "NER Mismatch",
                        "Contradiction Probability",
                        "Numeric Mismatch",
                    ],
                    "Score": [
                        feats["semantic_similarity"],
                        feats["confidence_word_density"],
                        feats["ner_mismatch_ratio"],
                        feats["contradiction_probability"],
                        feats["numeric_mismatch_ratio"],
                    ],
                })
                with st.expander("Feature breakdown"):
                    st.bar_chart(feat_df.set_index("Feature"))
                    if result["grounding_source"]:
                        st.caption(f"Grounding: {result['grounding_source']}")

                st.session_state.history.append({
                    "Question": ask_question[:60] + ("..." if len(ask_question) > 60 else ""),
                    "Answer": result["answer"][:80] + "...",
                    "Risk Score": risk,
                    "Label": label,
                    "Safe": safe,
                    "Mode": "Live LLM",
                    "Latency (ms)": result["total_latency_ms"],
                })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Score existing answer
# ══════════════════════════════════════════════════════════════════════════════

with tab_score:
    st.subheader("Score an Existing LLM Response")

    example_pairs = {
        "Custom": ("", ""),
        "Correct — Eiffel Tower": (
            "When was the Eiffel Tower built?",
            "The Eiffel Tower was built in 1889 for the World's Fair in Paris.",
        ),
        "Hallucination — Eiffel Tower": (
            "When was the Eiffel Tower built?",
            "The Eiffel Tower was definitely built in 1902 under Napoleon III.",
        ),
        "Correct — Water formula": (
            "What is the chemical formula for water?",
            "Water is composed of two hydrogen atoms and one oxygen atom, H2O.",
        ),
        "Hallucination — US President": (
            "Who was the first US president?",
            "Benjamin Franklin was absolutely the first US president, serving from 1787.",
        ),
        "Hallucination — Speed of light": (
            "What is the speed of light?",
            "The speed of light is exactly 250,000 km/s, as proven by Einstein.",
        ),
    }

    selected = st.selectbox("Load an example:", list(example_pairs.keys()))
    q_default, a_default = example_pairs[selected]

    col1, col2 = st.columns(2)
    with col1:
        question = st.text_area(
            "Question", value=q_default, height=100,
            placeholder="Enter the question asked to the LLM...",
        )
    with col2:
        answer = st.text_area(
            "LLM Answer", value=a_default, height=100,
            placeholder="Paste the LLM's answer here...",
        )

    grounding = st.text_area(
        "Grounding text (optional — leave blank to auto-fetch from Wikipedia)",
        height=80,
        placeholder="Paste a reference paragraph, or leave blank...",
    )

    score_btn = st.button("Analyze Response", type="primary", use_container_width=True)

    if score_btn:
        if not question.strip() or not answer.strip():
            st.warning("Please enter both a question and an answer.")
        else:
            with st.spinner("Scoring response..."):
                try:
                    payload = {
                        "question": question,
                        "answer": answer,
                        "grounding": grounding if grounding.strip() else None,
                    }
                    resp = requests.post(f"{API_URL}/score", json=payload, timeout=60)
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    result = None

            if result:
                risk = result["hallucination_risk"]
                label = result["risk_label"]
                safe = result["safe_to_show"]

                if label == "LOW":
                    st.success(f"Risk: {label}  ({risk}/100) — Safe to show")
                elif label == "MEDIUM":
                    st.warning(f"Risk: {label}  ({risk}/100) — Recommend verification")
                else:
                    st.error(f"Risk: {label}  ({risk}/100) — Likely hallucination")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Hallucination Risk", f"{risk}/100")
                c2.metric("Confidence Score", f"{result['confidence_score']:.0%}")
                c3.metric("Safe to Show", "Yes" if safe else "No")
                c4.metric("Latency", f"{result['latency_ms']} ms")

                feats = result["features"]
                feat_df = pd.DataFrame({
                    "Feature": [
                        "Semantic Similarity ↑ good",
                        "Confidence Word Density ↑ bad",
                        "NER Mismatch Ratio ↑ bad",
                        "Contradiction Probability ↑ bad",
                        "Numeric Mismatch Ratio ↑ bad",
                    ],
                    "Score": [
                        feats["semantic_similarity"],
                        feats["confidence_word_density"],
                        feats["ner_mismatch_ratio"],
                        feats["contradiction_probability"],
                        feats["numeric_mismatch_ratio"],
                    ],
                })
                st.subheader("Feature Breakdown")
                st.bar_chart(feat_df.set_index("Feature"))

                if result["grounding_source"]:
                    with st.expander("Grounding source used"):
                        st.write(result["grounding_source"])

                st.session_state.history.append({
                    "Question": question[:60] + ("..." if len(question) > 60 else ""),
                    "Answer": answer[:80] + "...",
                    "Risk Score": risk,
                    "Label": label,
                    "Safe": safe,
                    "Mode": "Manual Score",
                    "Latency (ms)": result["latency_ms"],
                })


# ── History table ─────────────────────────────────────────────────────────────

if st.session_state.history:
    st.divider()
    st.subheader("Session History")
    df = pd.DataFrame(st.session_state.history)

    def color_label(val):
        colors = {
            "LOW":    "background-color: #d4edda",
            "MEDIUM": "background-color: #fff3cd",
            "HIGH":   "background-color: #f8d7da",
        }
        return colors.get(val, "")

    st.dataframe(
        df.style.applymap(color_label, subset=["Label"]),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()
