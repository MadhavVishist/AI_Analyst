import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InsightStream | Enterprise AI Analyst",
    page_icon="🚀",
    layout="wide"
)

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #334155, #1e293b);
        color: #f8fafc;
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
    }
    h1, h2, h3 { color: #60a5fa !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_sidebar")
    
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
    elif "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
    
    if st.button("🗑️ Reset All"):
        st.session_state.clear()
        st.rerun()

# --- HELPER FUNCTIONS ---
def get_llm():
    if "GOOGLE_API_KEY" in os.environ:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return None

@st.cache_data
def robust_load_csv(file):
    for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    return None

def generate_eda_report(df):
    """Generates a quick statistical summary."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    eda_summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "numeric_cols": list(df.select_dtypes(include=['number']).columns),
        "categorical_cols": list(df.select_dtypes(include=['object']).columns)
    }
    return eda_summary

def export_chat_to_pdf():
    """Compiles chat history into a PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="InsightStream AI - Analysis Report", ln=True, align='C')
    pdf.ln(10)

    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "AI Analyst"
        clean_content = msg["content"].encode('latin-1', 'replace').decode('latin-1')
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{role}:", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 10, clean_content)
        pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin-1')

def create_agent(df):
    llm = get_llm()
    if not llm: return None
    
    # Simple history context
    history = ""
    if "messages" in st.session_state:
        for msg in st.session_state.messages[-4:]:
            history += f"{msg['role']}: {msg['content']}\n"
    
    prefix = f"""
    You are a Senior Data Scientist.
    History: {history}
    Instructions:
    1. Analyze the dataframe 'df'.
    2. If creating a plot, save as 'insight_plot.png'.
    3. Be concise and business-focused.
    """
    
    return create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
        prefix=prefix
    )

# --- MAIN APP LOGIC ---
st.title("🚀 InsightStream Enterprise")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar.")

elif uploaded_file:
    # 1. Load Data
    if "df" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        st.session_state.df = robust_load_csv(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []
    
    df = st.session_state.df

    # 2. Auto-EDA Panel (One-Click Report)
    with st.expander("📊 Data Health Dashboard (Auto-Generated)", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        eda = generate_eda_report(df)
        col1.metric("Rows", eda["rows"])
        col2.metric("Columns", eda["columns"])
        col3.metric("Missing Values", eda["missing"])
        col4.metric("Duplicates", eda["duplicates"])
        
        if st.checkbox("Show Correlation Heatmap"):
            if eda["numeric_cols"]:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(df[eda["numeric_cols"]].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else:
                st.info("No numeric columns for correlation.")

    # 3. Chat History
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg and msg["image"]:
                st.image(msg["image"])
    
    # 4. PDF Export Button
    if st.session_state.messages:
        pdf_bytes = export_chat_to_pdf()
        st.sidebar.download_button("📥 Export Report to PDF", data=pdf_bytes, file_name="analysis_report.pdf", mime="application/pdf")

    # 5. User Input
    if query := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.rerun()

    # 6. Processing (Handling the latest user message)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                agent = create_agent(df)
                if agent:
                    try:
                        last_query = st.session_state.messages[-1]["content"]
                        result = agent.invoke({"input": last_query})
                        response = result["output"]
                        
                        img_bytes = None
                        if os.path.exists("insight_plot.png"):
                            import io
                            with open("insight_plot.png", "rb") as f:
                                img_bytes = f.read()
                            st.image(img_bytes)
                            os.remove("insight_plot.png")
                        
                        st.markdown(response)
                        
                        # Append Assistant Response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "image": img_bytes
                        })
                        
                        # 7. Smart Follow-up Buttons (Simple Logic)
                        st.markdown("**Suggested Follow-ups:**")
                        cols = st.columns(3)
                        if cols[0].button("📈 Show Trends"):
                            st.session_state.messages.append({"role": "user", "content": "Analyze the trends over time."})
                            st.rerun()
                        if cols[1].button("💰 Analyze Profit"):
                            st.session_state.messages.append({"role": "user", "content": "What are the most profitable segments?"})
                            st.rerun()
                        if cols[2].button("⚠️ Find Outliers"):
                            st.session_state.messages.append({"role": "user", "content": "Identify any significant outliers in the data."})
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error: {e}")

else:
    st.info("👋 Upload a CSV file to begin.")