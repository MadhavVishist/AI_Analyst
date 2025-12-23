import streamlit as st
import pandas as pd
import os
import io

# LangChain & Gemini Integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="InsightStream | Autonomous Data Intelligence",
    page_icon="💎",
    layout="wide"
)

# Professional CSS: Animated Gradient Background & Glassmorphism
st.markdown("""
    <style>
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
    .glass-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 35px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 25px;
    }
    h1, h2, h3, p, label { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white; border: none; padding: 15px 30px;
        border-radius: 12px; font-weight: 700; width: 100%;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(37, 99, 235, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE LOGIC ---
def get_llm():
    # Priority: Sidebar Input -> Streamlit Secrets (for hosting)
    key = st.sidebar.text_input("Gemini API Key", type="password") or st.secrets.get("GOOGLE_API_KEY")
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        # Using the EXACT string confirmed by your debug script
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    return None

def robust_load_csv(file):
    for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    return None

# --- APP WORKFLOW ---
llm = get_llm()

st.title("💎 InsightStream AI")
st.markdown("##### Next-Gen Autonomous Strategy & Data Intelligence")

if not llm:
    st.warning("⚠️ Please provide your Gemini API Key in the sidebar to activate the AI Agent.")
else:
    # 1. Data Ingestion Panel
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Business Dataset (CSV)", type="csv")
    
    if uploaded_file:
        # Prevent re-loading same file
        if "df" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            st.session_state.df = robust_load_csv(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            # Initialize Agent
            st.session_state.agent = create_pandas_dataframe_agent(
                llm, st.session_state.df, verbose=True, allow_dangerous_code=True, 
                handle_parsing_errors=True, prefix="You are an expert BI Analyst. Save plots as 'insight_plot.png'."
            )
        
        st.success(f"Dataset Synced: {st.session_state.df.shape[0]} records loaded.")
        with st.expander("👁 View Data Stream"):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Strategic Analysis Panel
    if uploaded_file:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        query = st.text_area("What strategic insights do you need?", placeholder="e.g., 'Analyze sales trends and predict the top-performing category for next month.'")
        
        if st.button("🚀 INITIATE ANALYSIS") and query:
            with st.spinner("🧠 Agent is reasoning and calculating..."):
                try:
                    # Autonomous Step
                    result = st.session_state.agent.invoke({"input": query})
                    
                    # Formatting Step
                    prompt = ChatPromptTemplate.from_template("Reformat this analysis into a professional business report: {analysis}")
                    report = (prompt | llm | StrOutputParser()).invoke({"analysis": result["output"]})
                    
                    st.markdown("---")
                    st.markdown(report)
                    
                    if os.path.exists('insight_plot.png'):
                        st.image('insight_plot.png', caption="AI-Generated Strategic Evidence")
                        os.remove('insight_plot.png')
                except Exception as e:
                    st.error(f"Analysis Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)