"""
Clarity TTS: AI-Powered Support Triage Agent
Architecture: Streamlit UI + LangGraph Multi-Agent Workflow (OpenAI Engine)
"""

import streamlit as st
import time
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Clarity TTS Support Triage", layout="wide")

# ==========================================
# 2. LANGGRAPH STATE & AGENT ARCHITECTURE
# ==========================================
class AgentState(TypedDict):
    complaint_text: str
    category: str
    urgency: str
    key_entities: str
    agent_summary: str

def build_triage_graph(api_key: str):
    """Compiles the LangGraph Multi-Agent Workflow using OpenAI"""
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.0)

    def classifier_node(state: AgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an elite B2B airline support AI. Analyze the complaint and extract three data points. 
            You MUST output ONLY the three data points separated by pipes (|), with NO prefix labels.
            Format exactly like this: [Category] | [Low or Medium or High] | [Extracted PNR, Flight Number, Airline]
            Example output: API Ticketing Error | High | PNR: QWE123, Emirates
            CRITICAL: Use ONLY standard basic keyboard characters (ASCII). Do not use smart quotes or special symbols."""),
            ("human", "{complaint_text}")
        ])
        chain = prompt | llm
        response = chain.invoke({"complaint_text": state["complaint_text"]}).content
        parts = [p.strip() for p in response.split("|")]
        
        cat = parts[0].replace("Category:", "").strip() if len(parts) > 0 else "Uncategorized"
        urg = parts[1].replace("Urgency:", "").strip() if len(parts) > 1 else "Medium"
        ent = parts[2].replace("Entities:", "").strip() if len(parts) > 2 else "No specific entities found"
        
        return {"category": cat, "urgency": urg, "key_entities": ent}

    def summarizer_node(state: AgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for Clarity TTS B2B agents. Write a highly concise, 3-point summary of the core issue. Conclude with a 'Recommended Next Action' for the human support agent to execute immediately.
            CRITICAL INSTRUCTION: You MUST use standard keyboard dashes (-) for bullet points. Do NOT use special unicode bullet dots. Use ONLY standard ASCII characters. Do not use smart quotes."""),
            ("human", "Complaint: {complaint_text}\nAssigned Category: {category}\nAssigned Urgency: {urgency}")
        ])
        chain = prompt | llm
        response = chain.invoke({
            "complaint_text": state["complaint_text"],
            "category": state["category"],
            "urgency": state["urgency"]
        }).content
        return {"agent_summary": response}

    workflow = StateGraph(AgentState)
    workflow.add_node("classify", classifier_node)
    workflow.add_node("summarize", summarizer_node)
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "summarize")
    workflow.add_edge("summarize", END)
    
    return workflow.compile()

# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================

st.title("Clarity TTS | Intelligent Support Triage")
st.markdown("Automated B2B ticket classification and agentic summarization.")

# --- NEW EXPLANATION BLOCK ADDED HERE ---
st.info("This application resolves the operational bottleneck of manual support ticket processing. Powered by a multi-agent LangGraph architecture, it instantly reads incoming B2B complaints, evaluates their urgency, extracts critical entities (like PNRs), and generates a concise action plan for human agents. This reduces average handle time from minutes to seconds.")
# ----------------------------------------

st.divider()

# Ticket Database
sample_complaints = {
    "Select a sample B2B ticket...": "",
    "VIP GDS Sync Failure": "URGENT. Our VIP corporate client booked a first-class ticket to Tokyo (PNR: TYO998). The payment cleared on our end, but the booking is showing as 'Unconfirmed' in the Amadeus GDS. The flight leaves in 6 hours. If this isn't ticketed immediately, we will lose a million-dollar corporate account.",
    "Baggage Loss in Transit": "Our corporate client just landed in LHR on flight BA-112. Their specialized camera equipment is missing. The PNR is ZXC899. This is unacceptable, we need an immediate trace on this bag.",
    "NDC API Pricing Timeout": "We are getting a 504 Gateway Timeout error when calling the Clarity NDC Search API for Singapore Airlines routes. It works for one-way, but multi-city requests are hanging for 30+ seconds before timing out. Need tier-2 tech support to check the payload limits.",
    "Name Correction Request": "Hello, I made a typo on a booking for my client. PNR: QWE445. The passenger name is listed as 'Johnathan Doe' but his passport says 'Jonathan Doe' (no H). Delta is refusing to let him board without a formal name correction via the agency portal. Please override.",
    "Double Charge on Corporate Card": "I just booked a round trip to Dubai (PNR: DXB771). Your system crashed during checkout. I reloaded and booked it again, but my corporate card was charged twice for 85,000 INR. I need the duplicate charge reversed immediately.",
    "Standard Refund Request": "Customer cancelled their trip to Frankfurt due to illness. PNR is ABC123. Can you process the refund back to the original credit card as per the LOWRT fare rules?",
    "Corporate Portal Login Error": "Several of our travel agents cannot log into the Clarity B2B portal this morning. We are getting an 'Authentication Token Expired' error every time we hit the login endpoint. We have 40 bookings to process today, please escalate.",
    "Wheelchair Assistance Missing": "Passenger arriving at JFK on flight DL-404 (PNR: WHC332). We requested wheelchair assistance during the booking process through your portal, but the airline says the SSR code was never transmitted. Passenger has limited mobility, fix this before landing."
}

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("Control Panel")
    api_key = st.text_input("OpenAI API Key (sk-...)", type="password")
    st.divider()
    st.header("Incoming Ticket")
    preset_selection = st.selectbox("Load Ticket via API", list(sample_complaints.keys()))
    user_input = st.text_area("Ticket Body:", value=sample_complaints[preset_selection], height=250)
    
    analyze_btn = st.button("Run Agentic Triage", type="primary", use_container_width=True)
    st.divider()
    st.caption("Stack: Streamlit | LangGraph | GPT-4o-mini")

# ---------------------------------------------------------
# MAIN CANVAS
# ---------------------------------------------------------
if analyze_btn:
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not user_input.strip():
        st.warning("Please select or type a complaint ticket in the sidebar to analyze.")
    else:
        with st.status("Initializing Multi-Agent Workflow...", expanded=True) as status:
            st.write("Connecting to OpenAI Engine...")
            time.sleep(0.5)
            st.write("Step 1: Classifying domain and routing priority...")
            try:
                app_graph = build_triage_graph(api_key)
                initial_state = AgentState(complaint_text=user_input, category="", urgency="", key_entities="", agent_summary="")
                final_state = app_graph.invoke(initial_state)
                
                st.write("Step 2: Extracting PNR and flight entities...")
                time.sleep(0.5)
                st.write("Step 3: Synthesizing human-action summary...")
                time.sleep(0.5)
                status.update(label="Triage Complete!", state="complete", expanded=False)
                
                # Formatting urgency
                urgency_raw = final_state['urgency'].strip().upper()
                urgency_display = "HIGH" if "HIGH" in urgency_raw else "MEDIUM" if "MEDIUM" in urgency_raw else "LOW"
                
                st.markdown("### Ticket Metadata")
                col1, col2, col3 = st.columns(3)
                col1.info(f"**Urgency:**\n\n{urgency_display}")
                col2.info(f"**Category:**\n\n{final_state['category']}")
                col3.info(f"**Entities:**\n\n{final_state['key_entities']}")
                    
                st.divider()
                st.markdown("### Agent Action Summary")
                st.success(final_state['agent_summary'])
                
                with st.expander("View Full Raw Data"):
                    st.write(f"Original Text: {user_input}")
                
            except Exception as e:
                status.update(label="Execution Failed", state="error", expanded=True)
                st.error(f"Error: {str(e)}")
else:
    st.info("System idle. Select a ticket in the sidebar and initialize triage to view LangGraph output.")