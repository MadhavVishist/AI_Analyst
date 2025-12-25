import streamlit as st
import pandas as pd
import os

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Analyst",
    page_icon="💎",
    layout="wide"
)

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #020617, #0f172a, #1e1b4b, #312e81);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        margin-bottom: 10px;
    }
    .stChatMessage .stMarkdown {
        color: #f8fafc !important;
    }
    h1, h2, h3, p, label { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("Gemini API Key (optional)", type="password", key="gemini_api_key_sidebar")
    
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
    elif "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
    
    if st.button("🗑️ Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- HELPER FUNCTIONS ---
def get_llm():
    if "GOOGLE_API_KEY" in os.environ:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return None

def robust_load_csv(file):
    for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    return None

def get_history_context():
    if "messages" not in st.session_state: return ""
    context = ""
    for msg in st.session_state.messages[-6:]:
        role = "User" if msg["role"] == "user" else "AI"
        if "content" in msg:
            context += f"{role}: {msg['content']}\n"
    return context

def create_agent(df):
    llm = get_llm()
    if not llm: return None
    
    history = get_history_context()
    
    prefix = f"""
    You are an expert BI Analyst.
    
    ### CONVERSATION HISTORY:
    {history}
    
    ### INSTRUCTIONS:
    1. Answer the user's question using the dataframe 'df'.
    2. If the user says "it" or "previous", refer to the HISTORY above.
    3. If you generate a plot, save it as 'insight_plot.png'.
    4. IMPORTANT: You must return a final answer. Use the format:
       Final Answer: [Your answer here]
    """
    
    return create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        # --- FIXED: Correct way to handle parsing errors in newer versions ---
        agent_executor_kwargs={"handle_parsing_errors": True},
        prefix=prefix
    )

# --- MAIN APP LOGIC ---
st.title("💎 AI Data Analyst (business insights from your data)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar.")
elif uploaded_file:
    # Load Data
    if "df" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        st.session_state.df = robust_load_csv(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = [] 
    
    with st.expander("👁 View Raw Data Stream"):
        st.dataframe(st.session_state.df.head(5), use_container_width=True)

    # Render Chat
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg and msg["image"]:
                st.image(msg["image"])
                st.download_button("⬇️ Download Plot", msg["image"], file_name=f"plot_{i}.png", mime="image/png", key=f"btn_{i}")

    # User Input
    if query := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                agent = create_agent(st.session_state.df)
                if agent:
                    try:
                        result = agent.invoke({"input": query})
                        response = result["output"]
                        
                        img_bytes = None
                        if os.path.exists("insight_plot.png"):
                            with open("insight_plot.png", "rb") as f:
                                img_bytes = f.read()
                            st.image(img_bytes)
                            os.remove("insight_plot.png")
                        
                        st.markdown(response)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "image": img_bytes
                        })
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")

else:
    st.info("👋 Upload a CSV file to start.")