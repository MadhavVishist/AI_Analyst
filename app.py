import streamlit as st
import pandas as pd
import os
import io

# Langchain & Gemini Integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- ONE-PAGE UI CONFIGURATION ---
st.set_page_config(
    page_title="InsightStream | Autonomous Data Intelligence",
    page_icon="💎",
    layout="wide"
)

# Professional CSS for Animated Background and Glassmorphism
st.markdown("""
    <style>
    /* Animated Dynamic Background */
    .stApp {
        background: linear-gradient(-45deg, #020617, #0f172a, #1e1b4b, #312e81);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glass Container UI */
    .glass-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 35px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 25px;
    }

    /* Professional Typography */
    h1, h2, h3, p, label { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    
    /* Strategic Button */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 12px;
        font-weight: 700;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(37, 99, 235, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE INITIALIZATION ---
def get_llm():
    # Priority: Sidebar Input -> Streamlit Secrets
    key = st.sidebar.text_input("Enter Gemini API Key", type="password") or st.secrets.get("GOOGLE_API_KEY")
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        # Explicitly using gemini-2.5-flash as requested
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    return None

def robust_load_csv(file):
    """Handles various encodings to prevent UnicodeDecodeErrors."""
    for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    return None

def create_autonomous_agent(df, llm):
    prefix = """You are a world-class Business Intelligence Agent. 
    Always verify your logic using Python code (df.info(), df.head()).
    Save all visualizations as 'insight_plot.png'."""
    return create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True, 
        handle_parsing_errors=True, prefix=prefix
    )

def generate_business_report(analysis, llm):
    prompt = ChatPromptTemplate.from_template("""
        Convert the following data findings into a high-level executive report.
        Findings: {analysis}
        
        Sections:
        ## 📊 Executive Summary
        ## 🔍 Key Insights
        ## 💡 Strategic Recommendations
    """)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"analysis": analysis})

# --- UNIFIED DASHBOARD UI ---
llm = get_llm()

st.title("💎 InsightStream AI")
st.markdown("##### Next-Gen Autonomous Strategy & Data Intelligence")

if not llm:
    st.warning("⚠️ Please provide a Gemini API Key in the sidebar to activate the AI Agent.")
else:
    # --- SECTION 1: DATA INGESTION ---
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Business Dataset (CSV)", type="csv")
    
    if uploaded_file:
        if "df" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            st.session_state.df = robust_load_csv(uploaded_file)
            st.session_state.agent = create_autonomous_agent(st.session_state.df, llm)
            st.session_state.file_name = uploaded_file.name
        
        st.success(f"Intelligence Engine Synced: {st.session_state.df.shape[0]} rows ready.")
        with st.expander("👁 View Raw Data Stream"):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- SECTION 2: STRATEGIC ANALYSIS ---
    if uploaded_file:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        query = st.text_area("Define your strategic inquiry:", placeholder="e.g., 'Perform a cohort analysis on revenue and identify the highest growth regions.'")
        
        if st.button("🚀 INITIATE ANALYSIS"):
            with st.spinner("🧠 Agent is reasoning and calculating..."):
                try:
                    # Autonomous Step
                    result = st.session_state.agent.invoke({"input": query})
                    # Reporting Step
                    report = generate_business_report(result["output"], llm)
                    
                    st.markdown("---")
                    st.markdown(report)
                    
                    if os.path.exists('insight_plot.png'):
                        st.image('insight_plot.png', caption="AI-Generated Strategic Evidence")
                        os.remove('insight_plot.png')
                except Exception as e:
                    st.error(f"Reasoning Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)