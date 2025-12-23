import streamlit as st
import pandas as pd
import os
import io

# AI Orchestration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- ONE-PAGE UI CONFIG ---
st.set_page_config(
    page_title="InsightCore | Autonomous Data Analyst",
    page_icon="💎",
    layout="wide"
)

# Professional Live Animated Background & Glassmorphism CSS
st.markdown("""
    <style>
    /* Animated Dynamic Background */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #312e81, #1e3a8a);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glass Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Text & Input Styling */
    h1, h2, h3, p, label { color: #f8fafc !important; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    /* Strategic Button */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border: none;
        color: white;
        padding: 15px 32px;
        border-radius: 12px;
        font-weight: 700;
        width: 100%;
        transition: 0.3s all ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE INITIALIZATION ---
def get_llm():
    # Attempting to load key from secrets or sidebar
    key = st.sidebar.text_input("Gemini API Key", type="password") or st.secrets.get("GOOGLE_API_KEY")
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        # Use 'gemini-1.5-flash' stable ID to prevent 404
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    return None

def load_data(file):
    for enc in ['utf-8', 'latin1', 'cp1252']:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    return None

def create_agent(df, llm):
    prefix = "You are a Senior Strategic Analyst. Always use Python to verify facts. Save plots as 'insight_plot.png'."
    return create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True, 
        handle_parsing_errors=True, prefix=prefix
    )

def format_report(analysis, llm):
    prompt = ChatPromptTemplate.from_template("""
        Convert this data analysis into an executive strategy report.
        Analysis: {analysis}
        
        Use structure:
        ## 📊 Executive Summary
        ## 🔍 Key Findings
        ## 💡 Actionable Recommendations
    """)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"analysis": analysis})

# --- UI WORKFLOW ---
llm = get_llm()

st.title("💎 InsightCore AI")
st.markdown("##### Autonomous Strategic Intelligence for Global Datasets")

# One-Page Section 1: Data Upload
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your Business CSV", type=["csv"])
    if uploaded_file and llm:
        if "df" not in st.session_state or st.session_state.file_name != uploaded_file.name:
            st.session_state.df = load_data(uploaded_file)
            st.session_state.agent = create_agent(st.session_state.df, llm)
            st.session_state.file_name = uploaded_file.name
        
        st.success(f"Analysis Engine Ready: {st.session_state.df.shape[0]} records loaded.")
        with st.expander("👁 View Data Stream"):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# One-Page Section 2: Strategic Query
if uploaded_file and llm:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    query = st.text_area("What is your strategic goal?", placeholder="e.g., 'Identify the top 3 drivers of churn in the Q4 dataset.'")
    
    if st.button("🚀 EXECUTE ANALYSIS"):
        with st.spinner("🤖 Reasoning and executing code..."):
            try:
                # 1. Autonomous Agent Step
                result = st.session_state.agent.invoke({"input": query})
                # 2. Executive Formatting Step
                report = format_report(result["output"], llm)
                
                st.markdown("---")
                st.markdown(report)
                
                if os.path.exists('insight_plot.png'):
                    st.image('insight_plot.png', caption="Data Evidence Visualization")
                    os.remove('insight_plot.png')
            except Exception as e:
                st.error(f"Reasoning Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Please upload a CSV and provide an API Key to begin.")