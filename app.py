import streamlit as st
import pandas as pd
import os

# LangChain & Gemini Imports
# Note: We rely on the core libraries to avoid "ModuleNotFound" for memory
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
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
    }
    h1, h2, h3, p, label { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION (WIDGETS CREATED ONCE HERE) ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key Input (Fixed: Created only once)
    api_key = st.text_input("Gemini API Key", type="password", key="api_key_input")
    
    # 2. Handle Secrets Fallback
    if not api_key and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    
    # 3. Set Environment Variable
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.markdown("---")
    
    # 4. File Uploader
    uploaded_file = st.file_uploader("Upload CSV Data", type="csv")
    
    # 5. Reset Button
    if st.button("🗑️ Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- HELPER FUNCTIONS ---
def get_llm():
    """Returns the LLM instance only if key is set."""
    if "GOOGLE_API_KEY" in os.environ:
        # Using the stable model ID
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
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
    """Builds conversation history string for the agent."""
    if "messages" not in st.session_state: return ""
    context = ""
    # Use last 6 messages for context
    for msg in st.session_state.messages[-6:]:
        role = "User" if msg["role"] == "user" else "AI"
        # Only text, skip binary image data
        if "content" in msg:
            context += f"{role}: {msg['content']}\n"
    return context

def create_agent(df):
    """Creates the agent with manual history injection."""
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
    4. Be concise and business-focused.
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
    with st.expander("👁 View Dataset"):
        st.dataframe(st.session_state.df.head(5), width="stretch") # Fixed deprecated param

    # 1. Render Chat History
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render saved image if exists
            if "image" in msg and msg["image"]:
                st.image(msg["image"])
                st.download_button(
                    "⬇️ Download", 
                    msg["image"], 
                    file_name=f"plot_{i}.png", 
                    mime="image/png",
                    key=f"btn_{i}"
                )

    # 2. User Input Handling
    if query := st.chat_input("Ask a question about your data..."):
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate AI Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
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