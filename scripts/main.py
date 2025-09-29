import streamlit as st
import pandas as pd
import time
from data_loader import load_data
from agents import cost_analyzer, route_planner, compliance_checker, summary_agent, orchestrator

st.set_page_config(page_title="FreightOps AI Assistant", layout="wide")
st.title("🚚 FreightOps AI Assistant")

DATA_PATH = "data/freight_data.csv"

# --------------------------
# UI
# --------------------------
with st.expander("📊 Preview Data", expanded=False):
    try:
        df_preview = load_data(DATA_PATH)
        st.dataframe(df_preview.head(), width="stretch")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")

st.markdown("Use auto-routing or manually select an agent. The assistant runs analytics locally, then asks the LLM to explain the results.")

query = st.text_input("💬 Ask about freight operations:")
manual_agent = st.selectbox(
    "🔎 Choose Agent (or Auto)",
    ["Auto", "CostAnalyzer", "RoutePlanner", "ComplianceChecker", "SummaryAgent"],
    index=0
)


if st.button("Ask Assistant") and query:
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

    with st.spinner("Thinking..."):
        start = time.time()

        if manual_agent == "Auto":
            agent_name, fn = orchestrator(query)
        else:
            mapping = {
                "CostAnalyzer": ("Cost Analyzer", cost_analyzer),
                "RoutePlanner": ("Route Planner", route_planner),
                "ComplianceChecker": ("Compliance Checker", compliance_checker),
                "SummaryAgent": ("Summary Agent", summary_agent),
            }
            agent_name, fn = mapping[manual_agent]

        explanation, kpis, payload = fn(df, query)
        latency = round(time.time() - start, 2)

    st.success(f"✅ Response from {agent_name} (Latency: {latency}s)")
    st.write(explanation)

    # KPIs
    if kpis:
        cols = st.columns(min(4, len(kpis)))
        for i, k in enumerate(kpis):
            with cols[i % len(cols)]:
                st.metric(k["label"], k["value"])

    # Structured outputs
    if agent_name == "Summary Agent" and isinstance(payload, dict):
        st.subheader("📅 Monthly Summary")
        st.dataframe(payload["monthly"], width="stretch")
        st.subheader("📆 Weekly Summary")
        st.dataframe(payload["weekly"], width="stretch")
    else:
        # DataFrame payloads from other agents
        if isinstance(payload, pd.DataFrame) and not payload.empty:
            st.dataframe(payload, width="stretch")
        elif isinstance(payload, pd.DataFrame):
            st.info("No rows to display.")
        else:
            st.write(payload)
