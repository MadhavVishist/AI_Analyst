import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .report-box { background-color: #ffffff; padding: 25px; border-radius: 10px; border: 1px solid #e6e9ef; }
    </style>
    """, unsafe_allow_html=True)

# --- API KEY HANDLING ---
# Priority: 1. Streamlit Secrets (for hosting) -> 2. Sidebar Input
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if not api_key:
    api_key = st.secrets.get("GOOGLE_API_KEY")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# --- CORE LOGIC ---
def get_analyst_agent(df):
    # Uses the latest stable reasoning model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    return create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True
    )

def format_report(raw_text):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    template = ChatPromptTemplate.from_template("""
        Reformat this analysis into a professional business report:
        {analysis}
        
        Use sections: ## 📊 Executive Summary, ## 🔍 Key Findings, ## 📉 Data Evidence, ## ✅ Recommendations.
    """)
    chain = template | llm | StrOutputParser()
    return chain.invoke({"analysis": raw_text})

# --- UI INTERFACE ---
st.title("🤖 Autonomous Business Data Analyst")
st.write("Upload a CSV and ask any business question. The AI will write code and analyze it for you.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head(3), use_container_width=True)

    query = st.text_input("What is your business question?")

    if st.button("Analyze Data") and query:
        if not api_key:
            st.error("Please provide a Gemini API Key in the sidebar.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    agent = get_analyst_agent(df)
                    # Agent 'thinks' and runs code
                    result = agent.invoke({"input": query})
                    
                    # Formatter cleans the output
                    report = format_report(result["output"])
                    
                    st.markdown("---")
                    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                    
                    if os.path.exists("insight_plot.png"):
                        st.image("insight_plot.png")
                except Exception as e:
                    st.error(f"Error: {e}")