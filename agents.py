import logging
from typing import Dict, List, Any

from crewai import Agent
from langchain.chat_models import ChatOpenAI

from config import (
    OPENAI_API_KEY,
    DEFAULT_MODEL,
    MARKET_ANALYSIS_MODEL,
    RISK_SCORING_MODEL,
    PROJECT_TRACKING_MODEL,
    REPORTING_MODEL,
    TEMPERATURE
)
from tools import (
    langchain_tools,
    tools_dict
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize LLM models
default_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=TEMPERATURE,
    model_name=DEFAULT_MODEL
)

market_analysis_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=TEMPERATURE,
    model_name=MARKET_ANALYSIS_MODEL
)

risk_scoring_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=TEMPERATURE,
    model_name=RISK_SCORING_MODEL
)

project_tracking_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=TEMPERATURE,
    model_name=PROJECT_TRACKING_MODEL
)

reporting_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=TEMPERATURE,
    model_name=REPORTING_MODEL
)

# Create the Project Risk Manager agent
project_risk_manager = Agent(
    role="Project Risk Manager",
    goal="Identify, assess, and coordinate mitigation of all project risks",
    backstory="""
    You are a senior project risk manager with extensive experience in IT projects.
    Your expertise lies in identifying risks across multiple dimensions, assessing their 
    potential impact, and coordinating mitigation strategies. You have a holistic view 
    of projects and can integrate insights from various sources to create a comprehensive 
    risk management approach.
    """,
    verbose=True,
    llm=default_llm,
    tools=langchain_tools,
    allow_delegation=True
)

# Create the Market Analysis Agent
market_analysis_agent = Agent(
    role="Market Analysis Agent",
    goal="Analyze financial trends, market news, and economic indicators to identify external risks",
    backstory="""
    You are a market analysis specialist with deep knowledge of financial markets, 
    economic trends, and industry dynamics. You excel at identifying external factors 
    that could impact IT projects, such as market shifts, regulatory changes, and 
    economic conditions. Your analysis helps anticipate external risks before they 
    affect project outcomes.
    """,
    verbose=True,
    llm=market_analysis_llm,
    tools=[
        tools_dict["fetch_market_news_tool"],
        tools_dict["market_data_tool"],
        tools_dict["identify_external_risks_tool"]
    ],
    allow_delegation=False
)

# Create the Risk Scoring Agent
risk_scoring_agent = Agent(
    role="Risk Scoring Agent",
    goal="Evaluate and score identified risks based on probability, impact, and interdependencies",
    backstory="""
    You are a risk assessment expert specializing in quantitative risk analysis. 
    Your methodical approach to evaluating risk probability and impact enables 
    accurate risk scoring and prioritization. You can identify risk interdependencies 
    and calculate cumulative effects to determine overall project risk levels.
    """,
    verbose=True,
    llm=risk_scoring_llm,
    tools=[
        tools_dict["calculate_risk_score_tool"],
        tools_dict["project_risks_tool"],
        tools_dict["add_risk_tool"],
        tools_dict["update_risk_tool"]
    ],
    allow_delegation=False
)

# Create the Project Status Tracking Agent
project_status_agent = Agent(
    role="Project Status Tracking Agent",
    goal="Monitor project progress and identify internal risks related to resources, schedules, and technical aspects",
    backstory="""
    You are a project status tracking specialist with a keen eye for early warning 
    signs in IT projects. You can identify resource constraints, schedule slippages, 
    and technical hurdles before they escalate into major issues. Your focus is on 
    internal project dynamics and operational challenges that could introduce risks.
    """,
    verbose=True,
    llm=project_tracking_llm,
    tools=[
        tools_dict["project_info_tool"],
        tools_dict["analyze_project_health_tool"],
        tools_dict["analyze_risk_trends_tool"]
    ],
    allow_delegation=False
)

# Create the Reporting Agent
reporting_agent = Agent(
    role="Reporting Agent",
    goal="Generate detailed risk analytics, alerts, and reports for stakeholders",
    backstory="""
    You are a risk reporting specialist with expertise in translating complex risk 
    data into clear, actionable insights. You excel at creating comprehensive risk 
    reports that highlight critical issues, track risk trends, and provide mitigation 
    recommendations. Your reports enable informed decision-making by presenting risk 
    information in an accessible, prioritized format.
    """,
    verbose=True,
    llm=reporting_llm,
    tools=[
        tools_dict["generate_risk_report_tool"],
        tools_dict["get_project_reports_tool"],
        tools_dict["generate_mitigation_strategies_tool"]
    ],
    allow_delegation=False
)

# Create a dictionary of all agents
agents_dict = {
    "risk_manager": project_risk_manager,
    "market_analyst": market_analysis_agent,
    "risk_scorer": risk_scoring_agent,
    "project_tracker": project_status_agent,
    "reporting_agent": reporting_agent
}

# Create a list of all agents
agents_list = [
    project_risk_manager,
    market_analysis_agent,
    risk_scoring_agent,
    project_status_agent,
    reporting_agent
]
