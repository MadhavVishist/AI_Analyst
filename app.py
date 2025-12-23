import streamlit as st
import pandas as pd
import os
import io

# LangChain & Gemini Imports (No Memory Module needed)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType

# --- PAGE SETUP ---
st.set_page_config(
    page_title="InsightStream | Pro AI Analyst",
    page_icon="🧠",
    layout="wide"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
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
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
    }
    h1, h2, h3, p, label { color: #f8fafc !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC ---
def get_llm():
    # API Key Handling
    if "GOOGLE_API_KEY" not in os.environ:
        key = st.sidebar.text_input("Gemini API Key", type="password") or st.secrets.get("GOOGLE_API_KEY")
        if key:
            os.environ["GOOGLE_API_KEY"] = key
        else:
            return None
    # Using the stable model ID
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def robust_load_csv(file):
    # Fix for UnicodeDecodeError
    for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    return None

def get_chat_history_str():
    """
    Manually builds a history string to feed into the agent's prompt.
    This avoids importing the unstable Memory modules.
    """
    if "messages" not in st.session_state:
        return ""
    
    history_text = ""
    # We take the last 6 messages to keep context fresh but concise
    for msg in st.session_state.messages[-6:]:
        role = msg["role"].upper()
        content = msg["content"]
        history_text += f"{role}: {content}\n"
    return history_text

def create_agent(df):
    llm = get_llm()
    if not llm: return None

    # Inject History directly into the System Prompt
    history_context = get_chat_history_str()
    
    prefix = f"""
    You are an expert conversational Data Analyst.
    
    ### CONVERSATION HISTORY (Use this context for follow-up questions):
    {history_context}
    
    ### INSTRUCTIONS:
    1. Answer the user's current question based on the dataframe 'df'.
    2. If the user refers to "it" or "previous", check the CONVERSATION HISTORY above.
    3. If you create a plot, save it strictly as 'insight_plot.png'.
    4. Verify data types before plotting.
    """

    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        prefix=prefix
    )

# --- MAIN APP UI ---
st.title("🧠 InsightStream Pro")
st.caption("Autonomous AI Data Agent with Memory")

# Sidebar
with st.sidebar:
    st.header("🗂 Data Settings")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Logic
if uploaded_file:
    # Load Data (Once)
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.df = robust_load_csv(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = [] # Reset chat on new file

    # 1. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Restore Image if it exists in history
            if "image_data" in message and message["image_data"]:
                st.image(message["image_data"])
                st.download_button(
                    label="⬇️ Download Plot",
                    data=message["image_data"],
                    file_name=f"plot_{len(message['content'][:5])}.png",
                    mime="image/png",
                    key=f"hist_btn_{message['id']}"
                )

    # 2. Handle New Input
    if prompt := st.chat_input("Ask about your data..."):
        # Show User Message
        st.session_state.messages.append({"role": "user", "content": prompt, "id": len(st.session_state.messages)})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                agent = create_agent(st.session_state.df)
                if agent:
                    try:
                        # Run Agent
                        response = agent.invoke({"input": prompt})
                        output_text = response["output"]
                        
                        # Handle Image Logic
                        img_bytes = None
                        if os.path.exists("insight_plot.png"):
                            with open("insight_plot.png", "rb") as f:
                                img_bytes = f.read()
                            # Show Image
                            st.image(img_bytes)
                            st.download_button("⬇️ Download Plot", img_bytes, "insight_plot.png", "image/png")
                            os.remove("insight_plot.png") # Cleanup

                        st.markdown(output_text)

                        # Save to History
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": output_text,
                            "image_data": img_bytes,
                            "id": len(st.session_state.messages)
                        })
                        
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")
                else:
                    st.error("Please enter your API Key in the sidebar.")

else:
    st.info("👋 Please upload a CSV file to begin.")