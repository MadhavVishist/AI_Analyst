import streamlit as st
import pandas as pd
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PROFESSIONAL PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Data Analyst Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/MadhavVishist/AI_Analyst',
        'Report a bug': 'https://github.com/MadhavVishist/AI_Analyst/issues',
        'About': 'Professional AI-powered data analysis tool built with Streamlit and Google Gemini.'
    }
)

# --- ADVANCED CUSTOM STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #ffffff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
    }

    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e1e5e9;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 0.5rem;
    }

    .feature-desc {
        color: #718096;
        font-size: 0.9rem;
    }

    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(245,87,108,0.2);
    }

    .data-preview {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }

    .analysis-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(79,172,254,0.2);
    }

    .report-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border-left: 5px solid #667eea;
    }

    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }

    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .sidebar-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    .api-input {
        background: white;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        transition: border-color 0.3s ease;
    }

    .api-input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }

    .analyze-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin: 1rem 0;
    }

    .analyze-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.3);
    }

    .progress-bar {
        width: 100%;
        height: 4px;
        background: #e1e5e9;
        border-radius: 2px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
        animation: progress 2s ease-in-out infinite;
    }

    @keyframes progress {
        0% { width: 0%; }
        50% { width: 70%; }
        100% { width: 100%; }
    }

    .success-message {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }

    .error-message {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        border-top: 1px solid #e1e5e9;
        margin-top: 3rem;
    }

    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
    }

    .footer-link {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }

    .footer-link:hover {
        color: #764ba2;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR DESIGN ---
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-title">🔐 API Configuration</div>
            <div class="sidebar-subtitle">Secure your Gemini API access</div>
        </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Enter your Google Gemini API key...",
        help="Get your API key from Google AI Studio"
    )

    if not api_key:
        api_key = st.secrets.get("GOOGLE_API_KEY")

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("✅ API Key Configured")
    else:
        st.warning("⚠️ Please enter your API key to continue")

    st.markdown("---")

    st.markdown("""
        <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="margin: 0; color: #2d3748;">💡 Pro Tips</h4>
            <ul style="margin: 0.5rem 0; padding-left: 1.2rem; color: #4a5568; font-size: 0.9rem;">
                <li>Ask specific business questions</li>
                <li>Mention columns by name</li>
                <li>Request visualizations</li>
                <li>Use comparative analysis</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# --- MAIN HEADER ---
st.markdown("""
    <div class="main-header">
        <div class="main-title">AI Data Analyst Pro</div>
        <div class="subtitle">Transform your data into actionable business insights with advanced AI analysis</div>
    </div>
""", unsafe_allow_html=True)

# --- FEATURES SHOWCASE ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI-Powered Analysis</div>
            <div class="feature-desc">Advanced machine learning algorithms process your data intelligently</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Business Insights</div>
            <div class="feature-desc">Generate professional reports with actionable recommendations</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Instant Results</div>
            <div class="feature-desc">Get comprehensive analysis in seconds, not hours</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <div class="feature-title">Secure & Private</div>
            <div class="feature-desc">Your data stays local and secure throughout the analysis</div>
        </div>
    """, unsafe_allow_html=True)

# --- UPLOAD SECTION ---
st.markdown("""
    <div class="upload-section">
        <h2 style="margin: 0; font-size: 2rem;">📤 Upload Your Dataset</h2>
        <p style="margin: 0.5rem 0; opacity: 0.9;">Upload a CSV file to begin your AI-powered data analysis journey</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type="csv",
    help="Select a CSV file containing your business data for analysis"
)

# --- DATA PREVIEW ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("""
        <div class="data-preview">
            <h3 style="margin: 0 0 1rem 0; color: #2d3748;">📊 Data Overview</h3>
        </div>
    """, unsafe_allow_html=True)

    # Dataset stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Data Types", len(df.dtypes.unique()))
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{100-missing_pct:.1f}%")

    # Data preview
    st.dataframe(df.head(10), use_container_width=True)

    # Column analysis
    st.markdown("### 📋 Column Analysis")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null': df.count(),
        'Null %': ((df.isnull().sum() / len(df)) * 100).round(1)
    })
    st.dataframe(col_info, use_container_width=True)

    # --- ANALYSIS SECTION ---
    st.markdown("""
        <div class="analysis-section">
            <h2 style="margin: 0; font-size: 2rem;">🔍 Ask Your Business Question</h2>
            <p style="margin: 0.5rem 0; opacity: 0.9;">Describe what insights you want to uncover from your data</p>
        </div>
    """, unsafe_allow_html=True)

    query = st.text_area(
        "",
        placeholder="Example: What are the top 5 products by sales? Show me trends over time. Which customers are most valuable?",
        height=100,
        help="Be specific about what you want to analyze. Mention column names and desired insights."
    )

    # Analysis button
    if st.button("🚀 Generate Professional Analysis", type="primary", use_container_width=True):
        if not api_key:
            st.markdown("""
                <div class="error-message">
                    ⚠️ Please configure your Gemini API key in the sidebar to continue.
                </div>
            """, unsafe_allow_html=True)
        else:
            # Progress animation
            progress_placeholder = st.empty()
            progress_placeholder.markdown("""
                <div style="text-align: center; margin: 2rem 0;">
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                    <p style="color: #667eea; font-weight: 600; margin-top: 1rem;">Analyzing your data with AI...</p>
                </div>
            """, unsafe_allow_html=True)

            try:
                # Initialize AI agent
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
                agent = create_pandas_dataframe_agent(
                    llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True
                )

                # Perform analysis
                result = agent.invoke({"input": query})

                # Format report
                report_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
                template = ChatPromptTemplate.from_template("""
                    Transform this data analysis into a comprehensive professional business report:

                    ANALYSIS: {analysis}

                    Create a structured report with these sections:
                    ## 📊 Executive Summary
                    ## 🔍 Key Findings
                    ## 📈 Data Insights
                    ## 📉 Visual Evidence
                    ## ✅ Strategic Recommendations
                    ## 🎯 Action Items

                    Make it business-ready with clear language and actionable insights.
                """)
                chain = template | report_llm | StrOutputParser()
                report = chain.invoke({"analysis": result["output"]})

                # Clear progress
                progress_placeholder.empty()

                # Success message
                st.markdown("""
                    <div class="success-message">
                        ✅ Analysis Complete! Professional report generated below.
                    </div>
                """, unsafe_allow_html=True)

                # Display report
                st.markdown("""
                    <div class="report-container">
                """, unsafe_allow_html=True)

                st.markdown(report)

                # Check for generated plots
                if os.path.exists("insight_plot.png"):
                    st.markdown("### 📊 Generated Visualization")
                    st.image("insight_plot.png", use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # Download options
                st.markdown("### 💾 Export Options")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📄 Download Report",
                        report,
                        "analysis_report.md",
                        "text/markdown",
                        use_container_width=True
                    )
                with col2:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📊 Download Processed Data",
                        csv_data,
                        "analyzed_data.csv",
                        "text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                progress_placeholder.empty()
                st.markdown(f"""
                    <div class="error-message">
                        ❌ Analysis Error: {str(e)}
                        <br><small>Please check your query and data format, then try again.</small>
                    </div>
                """, unsafe_allow_html=True)

else:
    # Welcome message when no file uploaded
    st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f7fafc; border-radius: 15px; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #2d3748; margin-bottom: 1rem;">Ready to Transform Your Data?</h3>
            <p style="color: #718096; font-size: 1.1rem; margin: 0;">
                Upload a CSV file above to unlock powerful AI-driven business insights and professional reports.
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- PROFESSIONAL FOOTER ---
st.markdown("""
    <div class="footer">
        <div style="margin-bottom: 1rem;">
            <strong>AI Data Analyst Pro</strong> - Powered by Google Gemini & Streamlit
        </div>
        <div class="footer-links">
            <a href="https://github.com/MadhavVishist/AI_Analyst" class="footer-link" target="_blank">📖 Documentation</a>
            <a href="https://github.com/MadhavVishist/AI_Analyst/issues" class="footer-link" target="_blank">🐛 Report Issues</a>
            <a href="https://github.com/MadhavVishist" class="footer-link" target="_blank">👨‍💻 Developer</a>
        </div>
        <div style="margin-top: 1rem; font-size: 0.9rem; color: #a0aec0;">
            Built with ❤️ by Madhav Vishist • © 2025
        </div>
    </div>
""", unsafe_allow_html=True)