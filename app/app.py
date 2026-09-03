# app/app.py
# Planner-facing UI for the CPG demand forecasting agent.
#
# Three views:
#   Chat       — talks to the served DemandForecastingAgent
#   Forecasts  — reads gold.demand_predictions, the public track record table
#   Audit Log  — reads the governance tables, proving the "governed" claim
#                is real rather than asserted
#
# Auth: WorkspaceClient() auto-authenticates as the App's service principal
# when running inside Databricks Apps — no credentials handled in this file.

import os
import uuid
import json

import streamlit as st
import pandas as pd
from databricks.sdk import WorkspaceClient

st.set_page_config(page_title="CPG Demand Planner", layout="wide")

SERVING_ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "cpg-demand-agent")
WAREHOUSE_ID          = os.environ.get("WAREHOUSE_ID")

w = WorkspaceClient()


# ── Shared: SQL warehouse query helper ──────────────────────────────────────

@st.cache_data(ttl=60)
def run_query(statement: str) -> pd.DataFrame:
    """
    Runs a read-only query via the Statement Execution API against the
    warehouse bound in app.yaml. Cached for 60s so switching tabs doesn't
    re-hit the warehouse on every render.
    """
    if not WAREHOUSE_ID:
        return pd.DataFrame()

    result = w.statement_execution.execute_statement(
        warehouse_id = WAREHOUSE_ID,
        statement    = statement,
        wait_timeout = "30s",
    )

    if result.result is None or result.result.data_array is None:
        return pd.DataFrame()

    columns = [c.name for c in result.manifest.schema.columns]
    return pd.DataFrame(result.result.data_array, columns=columns)


def _extract_text(predictions) -> str:
    """
    ResponsesAgent output is a list of output items (message / function_call
    / function_call_output). We only want the assistant's final text.
    """
    try:
        output = predictions.get("output", []) if isinstance(predictions, dict) else predictions
        text_parts = []
        for item in output:
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
        return "".join(text_parts) or "No response text returned."
    except Exception:
        return "Couldn't parse the agent's response."


# ── Sidebar navigation ───────────────────────────────────────────────────────

view = st.sidebar.radio("View", ["Chat", "Forecasts", "Audit Log"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "First response after idle time can take 30-60s while the model "
    "endpoint scales up from zero. Later responses are fast."
)


# ── Chat view ─────────────────────────────────────────────────────────────

if view == "Chat":
    st.title("Ask the Demand Planning Agent")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input(
        "e.g. What are forecasted Food sales in Ontario next month?"
    )

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.write("Waking up the model — first query after idle "
                               "can take a minute...")
            try:
                response = w.serving_endpoints.query(
                    name   = SERVING_ENDPOINT_NAME,
                    inputs = {
                        "input": [{"role": "user", "content": prompt}],
                        "custom_inputs": {
                            "session_id": st.session_state.session_id
                        },
                    },
                )
                answer = _extract_text(response.predictions)
                placeholder.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                placeholder.error(
                    f"Couldn't reach the forecasting agent right now. "
                    f"({e})"
                )


# ── Forecasts view ───────────────────────────────────────────────────────

elif view == "Forecasts":
    st.title("Latest Forecasts")
    st.caption("Public track record — same table published in predictions/")

    df = run_query("""
        SELECT category_name, geo, ref_date, forecast_value,
               confidence_lower, confidence_upper, confidence_level
        FROM cpg_planning.gold.demand_predictions
        ORDER BY ref_date DESC, category_name, geo
        LIMIT 100
    """)

    if df.empty:
        st.info("No forecast data returned. Check WAREHOUSE_ID binding "
                "and that gold.demand_predictions has rows.")
    else:
        st.dataframe(df, use_container_width=True)


# ── Audit Log view ───────────────────────────────────────────────────────

elif view == "Audit Log":
    st.title("Agent Decision Audit Trail")
    st.caption(
        "Every agent invocation and tool call, logged in real time — "
        "this is what makes the system interrogable rather than a black box."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Decisions")
        decisions = run_query("""
            SELECT logged_at, agent_name, action, details
            FROM cpg_planning.governance.ai_decision_log
            ORDER BY logged_at DESC
            LIMIT 50
        """)
        if decisions.empty:
            st.info("No decisions logged yet.")
        else:
            st.dataframe(decisions, use_container_width=True)

    with col2:
        st.subheader("Tool Calls")
        tool_calls = run_query("""
            SELECT called_at, tool_name, parameters, result_summary
            FROM cpg_planning.governance.tool_call_history
            ORDER BY called_at DESC
            LIMIT 50
        """)
        if tool_calls.empty:
            st.info("No tool calls logged yet.")
        else:
            st.dataframe(tool_calls, use_container_width=True)
