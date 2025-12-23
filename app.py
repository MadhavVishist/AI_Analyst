import streamlit as st
import pandas as pd
import os
import io

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentType

# --- CONFIGURATION ---
st.set_page_config(
    page_title="InsightStream | Pro AI Analyst",
    page_icon="🧠",
    layout="wide"
)

# --- CUSTOM CSS ---
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
    /* Chat Bubble Styling */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    h1, h2, h3, p, label, .stMarkdown { color: #f8fafc !important; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZATION FUNCTIONS ---
def get_llm():
    if "GOOGLE_API_KEY" not in os.environ:
        # Check secrets or sidebar
        key = st.sidebar.text_input("Gemini API Key", type="password") or st.secrets.get("GOOGLE_API_KEY")
        if key:
            os.environ["GOOGLE_API_KEY"] = key
        else:
            return None
    # Use the model that worked for you
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def robust_load_csv(file):
    for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    return None

# --- AGENT FACTORY ---
def get_agent(df):
    """
    Initializes the agent with memory if it doesn't exist.
    We use a singleton pattern for the agent in session state
    to maintain its internal memory chain.
    """
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = ConversationBufferWindowMemory(
            k=10, # Remember last 10 interactions
            memory_key="chat_history",
            return_messages=True
        )
    
    llm = get_llm()
    if llm is None: return None

    # We add a specific instruction to the prompt to look at history
    prefix = """
    You are a conversational Data Analyst. 
    1. Answer the user's question based on the dataframe 'df'.
    2. If the user asks a follow-up question, use the CHAT_HISTORY context.
    3. If you create a plot, save it as 'insight_plot.png'.
    4. Always be concise and professional.
    """
    
    # We create a new agent but pass the PERSISTENT memory object
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        agent_type=AgentType.OPENAI_FUNCTIONS, # Robust tool calling for Gemini
        agent_executor_kwargs={"memory": st.session_state.agent_memory},
        prefix=prefix
    )

# --- MAIN APPLICATION ---
st.title("🧠 InsightStream Pro")
st.markdown("##### Conversational AI Data Analyst with Memory")

# 1. Sidebar Setup
with st.sidebar:
    st.header("🗂 Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.agent_memory.clear()
        st.rerun()

# 2. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Data Loading
if uploaded_file:
    # Load data only once per file upload
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.df = robust_load_csv(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        # Clear memory on new file
        st.session_state.messages = []
        if "agent_memory" in st.session_state: st.session_state.agent_memory.clear()

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Check if this message has an attached image
            if "image" in message and message["image"]:
                st.image(message["image"])
                st.download_button(
                    label="⬇️ Download Plot",
                    data=message["image"],
                    file_name=f"plot_{len(st.session_state.messages)}.png",
                    mime="image/png",
                    key=f"dl_{len(st.session_state.messages)}" # Unique key
                )

    # 4. Chat Input & Processing
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add User Message to History
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            agent = get_agent(st.session_state.df)
            if agent:
                with st.spinner("Thinking..."):
                    try:
                        # Invoke Agent
                        response = agent.invoke({"input": prompt})
                        output_text = response["output"]
                        
                        # Check for Image Generation
                        image_data = None
                        if os.path.exists("insight_plot.png"):
                            with open("insight_plot.png", "rb") as f:
                                image_data = f.read()
                            # Display immediately
                            st.image(image_data)
                            st.download_button(
                                label="⬇️ Download Plot",
                                data=image_data,
                                file_name="insight_plot.png",
                                mime="image/png"
                            )
                            # Clean up file system
                            os.remove("insight_plot.png")

                        st.markdown(output_text)

                        # Save Assistant Message to History (with image if exists)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": output_text,
                            "image": image_data
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Please provide an API Key.")

else:
    st.info("👋 Upload a CSV file to start the conversation.")