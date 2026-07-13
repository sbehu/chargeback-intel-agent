import streamlit as st
import os
import sys
from dotenv import load_dotenv

# app.py lives in chargeback-ui/, but orchestrator.py lives at the repo root.
# Streamlit adds the script's own directory to sys.path, not the repo root,
# so without this, the import below fails once deployed (works locally only
# if you happen to launch Streamlit from the repo root by chance).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import ChargebackOrchestrator

# Load fresh environment variables
load_dotenv()

# Page configurations
st.set_page_config(page_title="Chargeback Intel Platform", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🛡️ Chargeback Intel Platform")
st.sidebar.markdown("---")


@st.cache_resource
def get_orchestrator():
    """Initializes the pipeline once per server process (Pinecone connection,
    FlashRank reranker load, etc. are expensive to redo on every request)."""
    return ChargebackOrchestrator()


orchestrator = get_orchestrator()

# Initialize Session State Variables to prevent tab-switching data loss
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""
if "network" not in st.session_state:
    st.session_state.network = "Visa"
if "agent_output" not in st.session_state:
    st.session_state.agent_output = None
if "agent_debug" not in st.session_state:
    st.session_state.agent_debug = None


def run_chargeback_orchestrator(narration, card_network):
    """Runs the narrative through the real pipeline: query expansion ->
    hybrid Pinecone/BM25/FlashRank retrieval -> schema-constrained generation.
    Returns (formatted_output, raw_result_or_None) so the UI can show a
    debug view of the actual retrieved context alongside the rendered answer."""
    try:
        result = orchestrator.process_live_case(narration, card_network)

        if result["evidentiary_requirements"]:
            evidence_list = "\n".join(f"- {item}" for item in result["evidentiary_requirements"])
        else:
            evidence_list = "_Not applicable — no in-scope dispute evidence required._"

        formatted_output = (
            f"SYSTEM VERDICT: {result['verdict']}\n\n"
            f"**Cited Rule:** {result['cited_rule_id']}\n\n"
            f"**Resolution Narrative:**\n{result['defense_rationale']}\n\n"
            f"**Evidentiary Requirements:**\n{evidence_list}"
        )
        return formatted_output, result
    except Exception as e:
        return f"⚠️ **Backend Execution Error:** {str(e)}", None


# ==============================================================================
# AGENT PLAYGROUND
# ==============================================================================
st.title("💳 Chargeback Operations Center")
st.caption("Operational Interface for Risk Analysts • Production Environment")

st.markdown("### Process New Case Automation")

col_input, col_meta = st.columns([2, 1])
with col_input:
    # Read/Write directly from session state memory strings
    user_prompt = st.text_area(
        "Enter Dispute Details / Transaction Narration:",
        value=st.session_state.user_prompt,
        placeholder="e.g., Customer claims they never authorized the transaction for $120.00 at merchant store...",
        height=150
    )
    # Update state continuously
    st.session_state.user_prompt = user_prompt

    network_options = ["Visa", "Mastercard", "American Express"]
    default_index = network_options.index(st.session_state.network)
    network = st.selectbox("Card Network association:", network_options, index=default_index)
    st.session_state.network = network

    submit_btn = st.button("Run Intelligence Evaluation", type="primary")

with col_meta:
    st.info(
        "**Operational Note:**\n"
        "Submitting this process runs query expansion, hybrid Pinecone + BM25 retrieval, "
        "FlashRank re-ranking against the network rulebook index, and schema-constrained "
        "strategy generation, then returns a recommended resolution verdict."
    )

if submit_btn:
    if not user_prompt.strip():
        st.warning("Please enter transaction or dispute text details before running the evaluation.")
        st.session_state.agent_output = None
        st.session_state.agent_debug = None
    else:
        # High-performance hard keyword filter to catch out-of-domain strings instantly before LLM costs hit
        domain_keywords = ["charge", "dispute", "fraud", "merchant", "cardholder", "transaction", "billing", "delivery", "refund", "visa", "mastercard", "amex", "american express", "unauthorized", "stolen", "bought", "order", "price", "fee", "purchased", "item", "package"]
        user_input_lower = user_prompt.lower()
        is_valid_domain = any(keyword in user_input_lower for keyword in domain_keywords)

        if not is_valid_domain:
            st.session_state.agent_output = (
                "SYSTEM VERDICT: REJECTED\n"
                "Error: The provided narration does not appear to contain relevant transaction dispute metadata or industry vernacular."
            )
            st.session_state.agent_debug = None
        else:
            with st.spinner("Agent orchestrating retrieval and evaluation pipeline..."):
                formatted, raw = run_chargeback_orchestrator(user_prompt, network)
                st.session_state.agent_output = formatted
                st.session_state.agent_debug = raw

# Render data out of cache persistently if it exists
if st.session_state.agent_output:
    st.markdown("---")
    st.subheader("⚡ Automated System Resolution")

    parsed_verdict = "REVIEW"
    output_upper = st.session_state.agent_output.upper()
    if "VERDICT: ACCEPT" in output_upper:
        parsed_verdict = "ACCEPT"
    elif "VERDICT: CHALLENGE" in output_upper or "VERDICT: DENY" in output_upper:
        parsed_verdict = "DENY"
    elif "VERDICT: REJECTED" in output_upper:
        parsed_verdict = "REJECTED"

    v_col, d_col = st.columns([1, 4])
    with v_col:
        if parsed_verdict == "ACCEPT":
            st.metric(label="System Verdict", value="ACCEPT", delta="Dispute Accepted", delta_color="normal")
        elif parsed_verdict == "DENY":
            st.metric(label="System Verdict", value="CHALLENGE", delta="Merchant Defense Recommended", delta_color="inverse")
        elif parsed_verdict == "REJECTED":
            st.metric(label="System Verdict", value="REJECTED", delta="Out Of Domain Block", delta_color="inverse")
        else:
            st.metric(label="System Verdict", value="REVIEW", delta="Manual Escalation", delta_color="off")
    with d_col:
        st.markdown(st.session_state.agent_output)

    # --- DEBUG: show exactly what retrieval fed into generation ---
    if st.session_state.agent_debug:
        with st.expander("🔍 Retrieved Context (debug)"):
            st.markdown("**Expanded search query sent to Pinecone:**")
            st.code(st.session_state.agent_debug["enriched_query"], language="text")
            st.markdown("**Rule context actually retrieved and passed to generation:**")
            st.code(st.session_state.agent_debug["rule_context"], language="text")