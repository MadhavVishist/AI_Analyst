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
    /* Chat Message Styling */
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

# --- LOGIC ENGINE ---
def get_llm():
    key = st.sidebar.text_input("Gemini API Key", type="password") or st.secrets.get("GOOGLE_API_KEY")
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        # Using the specific model that worked for you
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
    """
    Manually creates a string of the conversation history.
    This replaces 'langchain.memory' to avoid import errors.
    """
    if "messages" not in st.session_state: return ""
    context = ""
    # Limit to last 6 turns to keep token usage efficient
    for msg in st.session_state.messages[-6:]:
        role = "User" if msg["role"] == "user" else "AI Analyst"
        # Only include text content in context, not image bytes
        context += f"{role}: {msg['content']}\n"
    return context

def create_conversational_agent(df):
    llm = get_llm()
    if not llm: return None
    
    # Inject History into Prompt
    history = get_history_context()
    
    prefix = f"""
    You are an expert BI Analyst and Strategic Advisor.
    
    ### CONVERSATION HISTORY (Use this for context):
    {history}
    
    ### INSTRUCTIONS:
    1. Answer the user's NEW question based on the dataframe 'df'.
    2. If the user refers to "it", "that", or "previous", use the HISTORY above.
    3. If asked to plot, save the chart as 'insight_plot.png'.
    4. Provide clear, business-oriented insights.
    """
    
    return create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True, 
        handle_parsing_errors=True, prefix=prefix
    )

# --- MAIN APP FLOW ---
llm = get_llm()

st.title("💎 InsightStream AI")
st.markdown("##### Next-Gen Autonomous Strategy & Data Intelligence")

# 1. Sidebar & Data Loading
with st.sidebar:
    st.header("🗂 Data Control")
    uploaded_file = st.file_uploader("Upload Business Dataset (CSV)", type="csv")
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Logic
if not llm:
    st.warning("⚠️ Please provide your Gemini API Key in the sidebar.")
elif uploaded_file:
    # Load Data Once
    if "df" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        st.session_state.df = robust_load_csv(uploaded_file)
        st.session_state.file_name = uploaded_file.name
        st.session_state.messages = [] # Reset chat on new file load
    
    # Display Data Preview (Collapsible)
    with st.expander("👁 View Raw Data Stream"):
        st.dataframe(st.session_state.df.head(5), use_container_width=True)

    # 2. Chat Interface (Loop through history)
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render Image if available in history
            if "image" in msg and msg["image"]:
                st.image(msg["image"], caption="AI Generated Insight")
                st.download_button(
                    label="⬇️ Download Plot",
                    data=msg["image"],
                    file_name=f"insight_plot_{i}.png",
                    mime="image/png",
                    key=f"dl_{i}"
                )

    # 3. User Input
    if query := st.chat_input("Ask a strategic question about your data..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Reasoning..."):
                agent = create_conversational_agent(st.session_state.df)
                if agent:
                    try:
                        # Invoke Agent
                        result = agent.invoke({"input": query})
                        response_text = result["output"]
                        
                        # IMAGE HANDLING LOGIC
                        img_bytes = None
                        if os.path.exists("insight_plot.png"):
                            # Read file into memory
                            with open("insight_plot.png", "rb") as f:
                                img_bytes = f.read()
                            
                            # Display immediately
                            st.image(img_bytes, caption="AI Generated Insight")
                            st.download_button("⬇️ Download Plot", img_bytes, "insight_plot.png", "image/png")
                            
                            # Clean up disk
                            os.remove("insight_plot.png")

                        # Display Text
                        st.markdown(response_text)

                        # SAVE TO HISTORY (Persist Image Bytes)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "image": img_bytes
                        })
                        
                    except Exception as e:
                        st.error(f"Reasoning Error: {e}")
else:
    st.info("👋 Upload a CSV file to activate the Autonomous Agent.")