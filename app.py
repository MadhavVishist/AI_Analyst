import streamlit as st
import pandas as pd
import os
import io
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Force Matplotlib to use non-interactive backend for server stability
matplotlib.use('Agg')

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InsightStream | Strategic AI Analyst",
    page_icon="📊",
    layout="wide"
)

# --- PROFESSIONAL UI CSS ---
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
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: scale(1.02);
    }
    h1, h2, h3 { color: #60a5fa !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR & CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_sidebar")
    
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
    elif "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
    
    if st.button("🗑️ Reset Analysis"):
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
    buffer = io.StringIO()
    df.info(buf=buffer)
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "numeric_cols": list(df.select_dtypes(include=['number']).columns)
    }

def export_chat_to_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="InsightStream Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "AI Analyst"
        # Clean text for PDF
        text = str(msg["content"]).encode('latin-1', 'replace').decode('latin-1')
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{role}:", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 10, text)
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

def create_agent(df):
    llm = get_llm()
    if not llm: return None
    
    # Context from previous messages
    history = ""
    if "messages" in st.session_state:
        for msg in st.session_state.messages[-4:]:
            history += f"{msg['role']}: {msg['content']}\n"
    
    # --- STRATEGIC BUSINESS PROMPT ---
    prefix = f"""
    You are a Senior Strategic Business Analyst. Your goal is to provide actionable insights, not just numbers.
    
    ### CONTEXT:
    {history}
    
    ### INSTRUCTIONS:
    1. **Analyze:** Use Python to query the dataframe 'df'.
    2. **Visualize:** If a trend, comparison, or distribution is useful, create a plot.
       - Use 'seaborn' or 'matplotlib'.
       - **CRITICAL:** Save the plot as 'insight_plot.png'. 
       - **DO NOT** use plt.show().
       - Ensure labels, titles, and legends are clear and professional.
    3. **Report:** Answer the user's question in this format:
       - **Executive Summary:** One sentence overview.
       - **Key Findings:** Bullet points with data evidence.
       - **Business Implication:** Why this matters.
    
    ### ERROR HANDLING:
    - If you cannot plot, explain why.
    - If data is missing, mention it.
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
st.title("📊 InsightStream: Strategic AI Analyst")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⚠️ Please enter your Gemini API Key to proceed.")
elif uploaded_file:
    # 1. Load Data
    if "df" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        st.session_state.df = robust_load_csv(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []
    
    df = st.session_state.df

    # 2. Auto-EDA Panel
    with st.expander("🔎 Data Health Snapshot", expanded=False):
        eda = generate_eda_report(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", eda["rows"])
        c2.metric("Columns", eda["columns"])
        c3.metric("Missing", eda["missing"])
        c4.metric("Duplicates", eda["duplicates"])

    # 3. Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg and msg["image"]:
                st.image(msg["image"])

    # 4. PDF Export
    if st.session_state.messages:
        try:
            pdf_data = export_chat_to_pdf()
            st.sidebar.download_button("📥 Download Report (PDF)", pdf_data, "analysis.pdf", "application/pdf")
        except: pass

    # 5. User Input
    if query := st.chat_input("Ask a strategic question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.rerun()

    # 6. Analysis Logic
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing business data..."):
                # Clear old plots to avoid duplicates
                plt.close('all')
                if os.path.exists("insight_plot.png"):
                    os.remove("insight_plot.png")
                
                agent = create_agent(df)
                if agent:
                    try:
                        # Run Agent
                        res = agent.invoke({"input": st.session_state.messages[-1]["content"]})
                        response_text = res["output"]
                        
                        # Capture Image if generated
                        img_bytes = None
                        if os.path.exists("insight_plot.png"):
                            with open("insight_plot.png", "rb") as f:
                                img_bytes = f.read()
                            st.image(img_bytes, caption="Generated Insight")
                        
                        st.markdown(response_text)
                        
                        # Save to History
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_text, 
                            "image": img_bytes
                        })
                        
                        # Suggested Follow-ups
                        st.markdown("---")
                        st.markdown("**Suggested Next Steps:**")
                        c1, c2, c3 = st.columns(3)
                        if c1.button("📈 Trends"):
                            st.session_state.messages.append({"role": "user", "content": "Show me the trends over time."})
                            st.rerun()
                        if c2.button("⚠️ Anomalies"):
                            st.session_state.messages.append({"role": "user", "content": "Are there any outliers?"})
                            st.rerun()
                        if c3.button("📊 Correlations"):
                            st.session_state.messages.append({"role": "user", "content": "What variables are correlated?"})
                            st.rerun()

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

else:
    st.info("👋 Upload a CSV file to begin.")