import streamlit as st
import os
import openai
from dotenv import load_dotenv

# Load fresh environment variables
load_dotenv()

# Page configurations
st.set_page_config(page_title="Chargeback Intel Platform", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🛡️ Chargeback Intel Platform")
st.sidebar.markdown("---")

# Initialize Session State Variables to prevent tab-switching data loss
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""
if "network" not in st.session_state:
    st.session_state.network = "Visa"
if "agent_output" not in st.session_state:
    st.session_state.agent_output = None

# Live backend orchestrator integration
def run_chargeback_orchestrator(narration, card_network):
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        system_prompt = (
            "You are a strict Chargeback and Fraud Intelligence Analyst Agent. Your ONLY job is to analyze "
            f"transaction disputes, credit card chargebacks, and fraud narratives for a {card_network} transaction. "
            "Cross-reference geographical risks, velocity rules, and compliance frameworks. Provide a clear "
            "'SYSTEM VERDICT' (either ACCEPT, REVIEW, or DENY) followed by a detailed bulleted 'Resolution Narrative' outlining your reasoning.\n\n"
            "CRITICAL GUARDRAIL:\n"
            "If the user's input is entirely unrelated to a credit card dispute, merchant transaction, fraud claim, "
            "or chargeback policy (such as general knowledge trivia, pop culture, math, or coding requests), you MUST "
            "strictly override your typical evaluation and output exactly this format:\n"
            "SYSTEM VERDICT: REJECTED\n"
            "Error: The provided narration is out-of-scope. The platform only processes transaction dispute records."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": narration}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ **Backend Execution Error:** {str(e)}"

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
        "Submitting this process runs the automated verification loops against compliance standards, "
        "cross-references historical Pinecone embedding indices, and outputs a recommended resolution verdict."
    )

if submit_btn:
    if not user_prompt.strip():
        st.warning("Please enter transaction or dispute text details before running the evaluation.")
        st.session_state.agent_output = None
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
        else:
            with st.spinner("Agent orchestrating data channels and evaluation arrays..."):
                st.session_state.agent_output = run_chargeback_orchestrator(user_prompt, network)

# Render data out of cache persistently if it exists
if st.session_state.agent_output:
    st.markdown("---")
    st.subheader("⚡ Automated System Resolution")
    
    parsed_verdict = "REVIEW"
    if "VERDICT: ACCEPT" in st.session_state.agent_output.upper() or "SYSTEM VERDICT: ACCEPT" in st.session_state.agent_output.upper():
        parsed_verdict = "ACCEPT"
    elif "VERDICT: DENY" in st.session_state.agent_output.upper() or "SYSTEM VERDICT: DENY" in st.session_state.agent_output.upper():
        parsed_verdict = "DENY"
    elif "VERDICT: REJECTED" in st.session_state.agent_output.upper() or "SYSTEM VERDICT: REJECTED" in st.session_state.agent_output.upper():
        parsed_verdict = "REJECTED"
    
    v_col, d_col = st.columns([1, 4])
    with v_col:
        if parsed_verdict == "ACCEPT":
            st.metric(label="System Verdict", value="ACCEPT", delta="Auto-Approved", delta_color="normal")
        elif parsed_verdict == "DENY":
            st.metric(label="System Verdict", value="DENY", delta="Auto-Rejected", delta_color="inverse")
        elif parsed_verdict == "REJECTED":
            st.metric(label="System Verdict", value="REJECTED", delta="Out Of Domain Block", delta_color="inverse")
        else:
            st.metric(label="System Verdict", value="REVIEW", delta="Manual Escalation", delta_color="off")
    with d_col:
        st.markdown(st.session_state.agent_output)