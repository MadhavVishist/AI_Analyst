import streamlit as st
import pandas as pd
import os
import io
import time

# Langchain & AI Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PROFESSIONAL UI CONFIGURATION ---
st.set_page_config(
    page_title="InsightStream | Autonomous BI Analyst",
    page_icon="💎",
    layout="wide"
)

# Custom CSS for Interactive Animated Mesh Background & Glassmorphism
st.markdown("""
    <style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(315deg, #1a2a6c, #b21f1f, #fdbb2d);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: white;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Containers */
    .report-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 35px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-top: 25px;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0, 198, 255, 0.4);
    }
    
    /* Input field visibility fixes */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SECURE API INITIALIZATION ---
def initialize_llm():
    key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    if not key:
        key = st.secrets.get("GOOGLE_API_KEY")
    
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        # Using the stable 'gemini-1.5-flash' ID to avoid 404s
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    return None

llm = initialize_llm()

# --- ROBUST DATA ENGINE ---
def load_data(file):
    for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    return None

def create_agent(df):
    prefix = """You are a Senior BI Analyst. 
    1. Inspect the data first (df.info()). 
    2. Save any charts as 'insight_plot.png'. 
    3. Use professional, data-driven reasoning."""
    
    return create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True, 
        handle_parsing_errors=True, prefix=prefix
    )

def generate_report(raw_analysis):
    template = ChatPromptTemplate.from_template("""
        Reformat this technical analysis into an executive-ready business report.
        Analysis: {analysis}
        
        Use this structure:
        # 📑 EXECUTIVE SUMMARY
        ## 🔍 CORE FINDINGS
        ## 📈 DATA EVIDENCE
        ## 💡 STRATEGIC ADVICE
    """)
    chain = template | llm | StrOutputParser()
    return chain.invoke({"analysis": raw_analysis})

# --- USER INTERFACE ---
with st.sidebar:
    st.markdown("# 💎 InsightStream")
    st.markdown("---")
    st.info("The next generation of autonomous data intelligence.")

st.title("Autonomous Data Intelligence")
st.markdown("### Upload your business datasets and extract strategic value in seconds.")

uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])

if uploaded_file and llm:
    if "df" not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
        with st.status("Analyzing Dataset Structure...", expanded=True) as status:
            st.session_state.df = load_data(uploaded_file)
            st.session_state.agent = create_agent(st.session_state.df)
            st.session_state.file_name = uploaded_file.name
            status.update(label="Analysis Ready", state="complete")

    tab1, tab2 = st.tabs(["📋 Data Explorer", "🤖 Intelligence Agent"])

    with tab1:
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    with tab2:
        query = st.text_area("Ask a strategic question:", placeholder="e.g., 'Identify the top 10% of customers by LTV and their common traits.'")
        
        if st.button("Generate Strategy Report"):
            with st.spinner("Processing deep reasoning..."):
                try:
                    result = st.session_state.agent.invoke({"input": query})
                    report = generate_report(result["output"])
                    
                    st.markdown(f'<div class="report-card">{report}</div>', unsafe_allow_html=True)
                    
                    if os.path.exists('insight_plot.png'):
                        st.image('insight_plot.png', caption="Strategic Visualization")
                        os.remove('insight_plot.png')
                except Exception as e:
                    st.error(f"Reasoning Failed: {e}")
elif not llm:
    st.warning("Please provide a valid API Key in the sidebar to activate the AI Agent.")