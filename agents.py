import os
from crewai import Agent, Crew, Task, Process
from config import LLM_TYPE, OLLAMA_MODEL, OLLAMA_BASE_URL
import json
from typing import List, Dict, Any, Optional

# Initialize the LLM
def get_llm():
    """Initialize and return the language model using Ollama."""
    from langchain_community.llms import Ollama
    
    # Configure Ollama LLM
    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0.2,
        base_url=OLLAMA_BASE_URL
    )
    return llm

# Define the agents
def create_project_risk_manager(llm) -> Agent:
    """Create the Project Risk Manager agent."""
    return Agent(
        role="Project Risk Manager",
        goal="Coordinate risk analysis and mitigation efforts across all agents to provide comprehensive project risk assessments",
        backstory="""You are an experienced project risk manager with expertise in IT projects. 
        Your responsibility is to coordinate the analysis from different specialized agents, 
        integrate their findings, and provide holistic risk assessments and mitigation strategies.
        You understand how different risk factors interact and can prioritize risks based on their 
        potential impact on project success.""",
        verbose=True,
        llm=llm,
        allow_delegation=True
    )

def create_market_analysis_agent(llm) -> Agent:
    """Create the Market Analysis Agent."""
    return Agent(
        role="Market Analysis Agent",
        goal="Analyze financial trends, market conditions, and news to identify external risks to IT projects",
        backstory="""You are a market analysis expert who specializes in identifying how 
        external market conditions can impact IT projects. You track industry trends, 
        competitor activities, economic indicators, and relevant news to detect potential 
        risks before they materialize. Your insights help teams prepare for market-driven challenges.""",
        verbose=True,
        llm=llm
    )

def create_risk_scoring_agent(llm) -> Agent:
    """Create the Risk Scoring Agent."""
    return Agent(
        role="Risk Scoring Agent",
        goal="Quantify and prioritize identified risks based on their probability and potential impact",
        backstory="""You are a risk assessment specialist with a background in statistical 
        analysis and risk modeling. You can evaluate both qualitative and quantitative data 
        to determine the severity of risks, their likelihood of occurrence, and their potential 
        impact on project outcomes. Your expertise helps teams focus on the most critical risks first.""",
        verbose=True,
        llm=llm
    )

def create_project_status_tracking_agent(llm) -> Agent:
    """Create the Project Status Tracking Agent."""
    return Agent(
        role="Project Status Tracking Agent",
        goal="Monitor internal project parameters and identify potential risks related to resources, schedules, and deliverables",
        backstory="""You are a project tracking expert who monitors the internal health of 
        IT projects. You analyze resource allocation, schedule adherence, budget consumption, 
        deliverable quality, and team dynamics to identify potential risks that could derail 
        project success. You are particularly adept at detecting early warning signs of project issues.""",
        verbose=True,
        llm=llm
    )

def create_reporting_agent(llm) -> Agent:
    """Create the Reporting Agent."""
    return Agent(
        role="Reporting Agent",
        goal="Generate comprehensive risk reports and alerts for decision-makers",
        backstory="""You are a communication specialist who excels at transforming complex 
        risk data into clear, actionable reports. You know how to prioritize information for 
        different stakeholders, highlight critical issues, and present mitigation options in 
        an accessible format. You ensure that decision-makers have the right information at 
        the right time to address project risks effectively.""",
        verbose=True,
        llm=llm
    )

# Initialize the crew with all agents
def initialize_crew() -> Crew:
    """Initialize and return the crew with all agents."""
    llm = get_llm()
    
    # Create all agents
    project_risk_manager = create_project_risk_manager(llm)
    market_analysis_agent = create_market_analysis_agent(llm)
    risk_scoring_agent = create_risk_scoring_agent(llm)
    project_status_tracking_agent = create_project_status_tracking_agent(llm)
    reporting_agent = create_reporting_agent(llm)
    
    # Create a crew with all agents
    crew = Crew(
        agents=[
            project_risk_manager,
            market_analysis_agent,
            risk_scoring_agent,
            project_status_tracking_agent,
            reporting_agent
        ],
        tasks=[],  # Tasks will be added dynamically based on user queries
        verbose=True,
        process=Process.sequential  # Agents will work sequentially
    )
    
    return crew

# Function to get project risk assessment based on user query
def get_project_risk_assessment(crew: Crew, user_query: str, selected_project: str) -> str:
    """
    Get a project risk assessment based on the user's query.
    
    Args:
        crew: The initialized crew of agents
        user_query: The user's question or request
        selected_project: The currently selected project or "All Projects"
        
    Returns:
        A response string with the risk assessment
    """
    from tasks import (
        create_analyze_market_conditions_task,
        create_assess_project_status_task,
        create_generate_risk_assessment_task,
        create_score_project_risks_task,
        create_generate_risk_report_task
    )
    
    # Clear any existing tasks
    crew.tasks = []
    
    # Create the market analysis task
    market_analysis_task = create_analyze_market_conditions_task(
        crew.agents[1],  # Market Analysis Agent
        user_query,
        selected_project
    )
    
    # Create the project status tracking task
    project_status_task = create_assess_project_status_task(
        crew.agents[3],  # Project Status Tracking Agent
        user_query,
        selected_project
    )
    
    # Create the risk scoring task
    risk_scoring_task = create_score_project_risks_task(
        crew.agents[2],  # Risk Scoring Agent
        user_query,
        selected_project,
        [market_analysis_task, project_status_task]
    )
    
    # Create the risk assessment task
    risk_assessment_task = create_generate_risk_assessment_task(
        crew.agents[0],  # Project Risk Manager
        user_query,
        selected_project,
        [market_analysis_task, project_status_task, risk_scoring_task]
    )
    
    # Create the reporting task
    reporting_task = create_generate_risk_report_task(
        crew.agents[4],  # Reporting Agent
        user_query,
        selected_project,
        [risk_assessment_task]
    )
    
    # Add all tasks to the crew
    crew.tasks = [
        market_analysis_task,
        project_status_task,
        risk_scoring_task,
        risk_assessment_task,
        reporting_task
    ]
    
    # Execute the tasks and get the result
    result = crew.kickoff()
    
    # Return the final report from the reporting agent
    return result
