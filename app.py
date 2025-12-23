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
    page_title="InsightStream | AI Analyst",
    page_icon="💎",
    layout="wide"
)

# --- PROFESSIONAL CSS (From your working snippet) ---
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

# --- SIDEBAR CONFIGURATION (CRITICAL FIX FOR DUPLICATE ID) ---
# We define the widget ONCE here to avoid 'DuplicateElementId' errors
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key Input
    api_key_input = st.text_input("Gemini API Key", type="password", key="gemini_api_key_sidebar")
    
    # 2. Logic to set key
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
    elif "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    st.markdown("---")
    
    # 3. File Uploader
    uploaded_file = st.file_uploader("Upload Business Dataset (CSV)", type="csv")
    
    # 4. Reset Button
    if st.button("🗑️ Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- HELPER FUNCTIONS ---
def get_llm():
    """
    Returns the LLM instance using the EXACT model string that works for you.
    """
    if "GOOGLE_API_KEY" in os.environ:
        # CRITICAL UPDATE: Using gemini-2.5-flash as requested
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return None

def robust_load_csv(file):
    """Loads CSV with fallback encodings."""
    for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    return None

def get_history_context():
    """
    Manually builds conversation history string for the agent.
    This enables 'Multi-Question' memory without crashing Streamlit.
    """
    if "messages" not in st.session_state: return ""
    context = ""
    # Use last 6 messages for context
    for msg in st.session_state.messages[-6:]:
        role = "User" if msg["role"] == "user" else "AI"
        if "content" in msg:
            context += f"{role}: {msg['content']}\n"
    return context

def create_agent(df):
    """Creates the agent with manual history injection."""
    llm = get_llm()
    if not llm: return None
    
    history = get_history_context()
    
    prefix = f"""
    You are an expert BI Analyst and Strategic Advisor.
    
    ### CONVERSATION HISTORY (Use this for context on "previous" questions):
    {history}
    
    ### INSTRUCTIONS:
    1. Answer the user's NEW question using the dataframe 'df'.
    2. If the user refers to "it", "that", or "previous", use the HISTORY above.
    3. If you generate a plot, save it strictly as 'insight_plot.png'.
    4. Provide clear, business-oriented insights.
    """
    
    return create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True, 
        handle_parsing_errors=True, 
        prefix=prefix
    )

# --- MAIN APP LOGIC ---
st.title("💎 InsightStream AI")
st.markdown("##### Next-Gen Autonomous Strategy & Data Intelligence")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main Workflow
if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar.")
elif uploaded_file:
    # Load Data Logic
    if "df" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        st.session_state.df = robust_load_csv(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = [] # Reset on new file
    
    # Display Data Preview
    with st.expander("👁 View Raw Data Stream"):
        st.dataframe(st.session_state.df.head(5), use_container_width=True)

    # 1. Render Chat History
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render saved image if exists
            if "image" in msg and msg["image"]:
                st.image(msg["image"])
                st.download_button(
                    "⬇️ Download Plot", 
                    msg["image"], 
                    file_name=f"plot_{i}.png", 
                    mime="image/png",
                    key=f"btn_{i}"
                )

    # 2. User Input Handling (Chat Interface)
    if query := st.chat_input("Ask a strategic question about your data..."):
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate AI Response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Reasoning..."):
                agent = create_agent(st.session_state.df)
                if agent:
                    try:
                        # Invoke Agent
                        result = agent.invoke({"input": query})
                        response = result["output"]
                        
                        # Handle Image Capture
                        img_bytes = None
                        if os.path.exists("insight_plot.png"):
                            with open("insight_plot.png", "rb") as f:
                                img_bytes = f.read()
                            st.image(img_bytes)
                            st.download_button("⬇️ Download Plot", img_bytes, "insight_plot.png", "image/png")
                            os.remove("insight_plot.png")
                        
                        st.markdown(response)
                        
                        # Save to History
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "image": img_bytes
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

else:
    st.info("👋 Upload a CSV file to start.")