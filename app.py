import streamlit as st
import pandas as pd
import os
import io

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PROFESSIONAL UI CONFIG ---
st.set_page_config(
    page_title="InsightCore | Autonomous Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look & Feel
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #F8F9FB; }
    
    /* Custom Card Design */
    .report-container {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border-left: 5px solid #007BFF;
    }
    
    /* Heading Styles */
    h1 { color: #1E293B; font-weight: 800 !important; }
    h2 { color: #334155; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px; }
    
    /* Button Hover Effect */
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: AUTH & SETTINGS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("InsightCore AI")
    st.markdown("---")
    
    # API Key Management
    st.subheader("🔑 Security")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter your key...")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key loaded from secrets.")
    
    st.markdown("---")
    st.info("💡 **Pro Tip:** Ask about trends, anomalies, or year-over-year growth for the best insights.")

# --- CORE AGENT LOGIC ---
def load_data_robustly(file):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for enc in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    return None

def create_analyst_agent(df):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    prefix = """
    You are a world-class Business Intelligence Analyst. 
    1. Reason through the data carefully.
    2. Use professional visualizations where possible.
    3. Always save any plot as 'insight_plot.png'.
    """
    return create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True, 
        handle_parsing_errors=True, prefix=prefix
    )

def format_business_report(raw_analysis):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    template = ChatPromptTemplate.from_template("""
        As a Senior Strategy Consultant, transform this technical output into a high-level executive report.
        Analysis Context: {analysis}
        
        Required Sections:
        ## 📊 Executive Summary
        ## 🔍 Key Findings
        ## 💡 Strategic Recommendations
    """)
    chain = template | llm | StrOutputParser()
    return chain.invoke({"analysis": raw_analysis})

# --- MAIN WORKSPACE ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Autonomous Data Analyst")
    st.markdown("#### Transforming raw datasets into strategic decisions.")

# 1. File Upload Section
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    # Handle Session State to avoid unnecessary re-computation
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.status("📁 Processing data...", expanded=True) as status:
            st.session_state.df = load_data_robustly(uploaded_file)
            if st.session_state.df is not None:
                st.session_state.agent = create_analyst_agent(st.session_state.df)
                st.session_state.current_file = uploaded_file.name
                status.update(label="✅ Data Ready for Analysis", state="complete", expanded=False)
            else:
                st.error("Encoding error: File could not be read.")

    # 2. Data Overview Tabs
    tab1, tab2 = st.tabs(["📋 Dataset Preview", "📈 Statistics Summary"])
    with tab1:
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
    with tab2:
        st.write(st.session_state.df.describe())

    # 3. Query Interface
    st.markdown("### 💬 Ask your Data a Question")
    user_query = st.text_input("", placeholder="e.g., 'What is the correlation between sales and profit margin across product lines?'")

    if st.button("Generate Executive Analysis"):
        if not user_query:
            st.warning("Please enter a question first.")
        else:
            with st.spinner("🧠 AI Agent is performing deep analysis..."):
                try:
                    # Agent Loop
                    result = st.session_state.agent.invoke({"input": user_query})
                    
                    # Formatting Loop
                    report = format_business_report(result["output"])
                    
                    # Output Display
                    st.markdown("---")
                    st.markdown(f'<div class="report-container">{report}</div>', unsafe_allow_html=True)
                    
                    # Visualization Display
                    if os.path.exists('insight_plot.png'):
                        st.markdown("### 📈 Visual Evidence")
                        st.image('insight_plot.png', use_container_width=True)
                        os.remove('insight_plot.png')
                        
                except Exception as e:
                    st.error(f"Analysis interrupted: {e}")

else:
    st.info("👋 Welcome! Please upload a CSV file in the section above to start your analysis.")