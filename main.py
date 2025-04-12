import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from agents import initialize_crew, get_project_risk_assessment
from config import RISK_LEVELS, RISK_CATEGORIES, DEFAULT_PROJECTS, VECTOR_DB_TYPE
from utils import format_chat_history, generate_risk_report_summary
from data_handlers import (
    get_project_data, 
    load_chat_history, 
    save_chat_history, 
    initialize_vector_db,
    populate_vector_db_with_sample_data,
    query_risks_from_vector_db
)

# Set page configuration
st.set_page_config(
    page_title="AI Project Risk Management System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "current_project" not in st.session_state:
    st.session_state.current_project = "All Projects"
if "crew" not in st.session_state:
    try:
        st.session_state.crew = initialize_crew()
    except Exception as e:
        st.error(f"Error initializing AI agents: {str(e)}")
        st.session_state.crew = None
if "vector_db" not in st.session_state:
    st.session_state.vector_db = initialize_vector_db()
    if not st.session_state.vector_db:
        st.warning(f"Vector database ({VECTOR_DB_TYPE}) initialization failed. Some search functionality may be limited.")
    else:
        # Populate with sample data only if newly initialized
        with st.spinner("Initializing risk database..."):
            populate_vector_db_with_sample_data()

# Main title and introduction
st.title("🔍 AI-Powered Project Risk Management System")
st.markdown("""
This system continuously monitors your IT projects to identify, assess, and help mitigate risks in real-time.
Our AI agents analyze internal project parameters and external market conditions to provide actionable insights.
""")

# Sidebar for project selection and filters
with st.sidebar:
    st.header("Project Navigator")
    selected_project = st.selectbox(
        "Select Project",
        ["All Projects"] + DEFAULT_PROJECTS,
        index=0,
        key="project_selector"
    )
    
    if selected_project != st.session_state.current_project:
        st.session_state.current_project = selected_project
    
    st.divider()
    
    # Time range for data
    st.subheader("Time Range")
    days_back = st.slider("Days of historical data", 7, 90, 30)
    
    # Risk filter
    st.subheader("Risk Filters")
    selected_risk_levels = st.multiselect(
        "Risk Levels",
        options=list(RISK_LEVELS.keys()),
        default=list(RISK_LEVELS.keys())
    )
    
    selected_categories = st.multiselect(
        "Risk Categories",
        options=RISK_CATEGORIES,
        default=RISK_CATEGORIES
    )
    
    # Refresh button
    if st.button("Refresh Analysis", type="primary"):
        st.toast("Refreshing risk analysis...", icon="🔄")
        # This will trigger a state refresh

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Dashboard", "Risk Analysis", "Chat Assistant"])

# Dashboard Tab
with tab1:
    st.header(f"Dashboard: {selected_project}")
    
    # Get project data based on selection
    try:
        project_data = get_project_data(selected_project, days_back)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_risks = len(project_data["risks"])
            st.metric("Total Risks", total_risks)
        
        with col2:
            high_risks = len([r for r in project_data["risks"] if r["level"] == "High"])
            st.metric("High Risks", high_risks)
        
        with col3:
            risk_trend = project_data["risk_trend"]
            st.metric("Risk Trend", f"{risk_trend:+.1f}%", 
                     delta_color="inverse" if risk_trend < 0 else "normal")
        
        with col4:
            mitigation_rate = project_data["mitigation_rate"]
            st.metric("Mitigation Rate", f"{mitigation_rate:.1f}%")
        
        # Risk trend chart
        st.subheader("Risk Trend Over Time")
        fig = px.line(
            project_data["trend_data"], 
            x="date", 
            y="risk_score",
            title="Overall Risk Score Trend",
            labels={"date": "Date", "risk_score": "Risk Score"}
        )
        st.plotly_chart(fig, use_container_width=True, key="trend_chart")
        
        # Risk distribution chart
        st.subheader("Risk Distribution by Category")
        risk_by_category = pd.DataFrame(project_data["risk_by_category"])
        fig2 = px.bar(
            risk_by_category,
            x="category",
            y="count",
            color="level",
            title="Risks by Category and Severity",
            labels={"category": "Risk Category", "count": "Number of Risks"},
            color_discrete_map={"Low": "#26eb77", "Medium": "#f0cc45", "High": "#eb4034"}
        )
        st.plotly_chart(fig2, use_container_width=True, key="category_chart")
        
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")

# Risk Analysis Tab
with tab2:
    st.header(f"Risk Analysis: {selected_project}")
    
    # Add semantic search function
    st.subheader("Risk Search")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input("Search risks by description or category", placeholder="e.g., resource shortage, security vulnerability")
    with search_col2:
        search_button = st.button("Search", type="primary", key="search_risks_button")
    
    try:
        # Get project risks
        project_risks = project_data["risks"]
        
        # If a search query was entered
        if search_query and search_button:
            with st.spinner("Searching risks database..."):
                # Search risks in vector DB
                if st.session_state.vector_db:
                    vector_results = query_risks_from_vector_db(
                        search_query, 
                        project=selected_project if selected_project != "All Projects" else None,
                        limit=10
                    )
                    if vector_results:
                        st.success(f"Found {len(vector_results)} matching risks")
                        project_risks = vector_results
                    else:
                        st.info("No matching risks found in the database. Showing all risks instead.")
        
        # Filter risks based on sidebar selections
        filtered_risks = [
            risk for risk in project_risks 
            if risk["level"] in selected_risk_levels and risk["category"] in selected_categories
        ]
        
        if not filtered_risks:
            st.info("No risks match your current filters.")
        else:
            # Display risks in expandable sections
            for i, risk in enumerate(filtered_risks):
                with st.expander(f"{risk['level']} Risk: {risk['title']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {risk['description']}")
                        st.markdown(f"**Category:** {risk['category']}")
                        st.markdown(f"**Impact:** {risk['impact']}")
                        st.markdown(f"**Probability:** {risk['probability']}")
                        
                        # Mitigation strategies
                        st.subheader("Mitigation Strategies")
                        for idx, strategy in enumerate(risk['mitigation_strategies']):
                            st.markdown(f"{idx + 1}. {strategy}")
                    
                    with col2:
                        # Risk score gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = risk['score'],
                            title = {'text': "Risk Score"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkgrey"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': risk['score']
                                }
                            }
                        ))
                        fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(fig, use_container_width=True, key=f"risk_gauge_{i}")
            
            # Risk report summary
            st.subheader("Risk Report Summary")
            report_summary = generate_risk_report_summary(selected_project, filtered_risks)
            st.markdown(report_summary)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "Download Risk Report (CSV)",
                    data=pd.DataFrame(filtered_risks).to_csv(index=False),
                    file_name=f"risk_report_{selected_project.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col2:
                if st.button("Generate Detailed PDF Report"):
                    st.info("PDF report generation would be implemented here")
            with col3:
                vector_db_type = "ChromaDB" if VECTOR_DB_TYPE == "chromadb" else "Pinecone"
                st.info(f"Risk data stored in {vector_db_type}")
        
    except Exception as e:
        st.error(f"Error loading risk analysis: {str(e)}")

# Chat Assistant Tab
with tab3:
    st.header("Project Risk Chat Assistant")
    st.markdown("""
    Chat with our AI assistant to get real-time information about project risks, 
    status updates, and mitigation strategies. Ask questions like:
    - What are the top risks for Project X?
    - How has the risk profile changed in the last week?
    - What mitigation strategies do you recommend for resource shortages?
    - Compare risks between Project A and Project B
    """)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about project risks...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing projects and risks..."):
                try:
                    if st.session_state.crew:
                        response = get_project_risk_assessment(
                            st.session_state.crew, 
                            user_input, 
                            selected_project
                        )
                    else:
                        response = "I'm having trouble connecting to the risk analysis system. Please try again later."
                except Exception as e:
                    response = f"I encountered an error while analyzing your request: {str(e)}"
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Save chat history
        save_chat_history(st.session_state.chat_history)
        
        # Rerun to update the chat display
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888888; font-size: 0.8em;">
    AI-Powered Project Risk Management System | Last updated: {}
</div>
""".format(datetime.now().strftime("%B %d, %Y %H:%M")), unsafe_allow_html=True)
