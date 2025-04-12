import json
import logging
from typing import Dict, List, Optional, Any

from crewai import Task
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

def create_project_analysis_task(agent):
    """Create project analysis task for the Project Risk Manager agent."""
    return Task(
        description="""
        Analyze a project to identify potential risks across multiple dimensions.
        1. Get project details and understand its scope, timeline, and objectives
        2. Examine existing risks and identify gaps
        3. Assess both internal and external risk factors
        4. Prepare a comprehensive risk analysis
        
        Input: The project ID to analyze
        Output: A comprehensive risk analysis including identified risks, overall risk score, and initial recommendations
        """,
        expected_output="""
        A JSON object containing:
        1. Project details summary
        2. Comprehensive list of identified risks
        3. Overall risk assessment
        4. Initial recommendations for risk mitigation
        """,
        agent=agent,
        tools=[
            tools_dict["project_info_tool"],
            tools_dict["project_risks_tool"],
            tools_dict["calculate_risk_score_tool"],
            tools_dict["market_data_tool"],
            tools_dict["analyze_project_health_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are conducting a comprehensive risk analysis for an IT project. Be thorough and identify both obvious and non-obvious risks."}
        ]
    )

def create_market_analysis_task(agent):
    """Create market analysis task for the Market Analysis Agent."""
    return Task(
        description="""
        Analyze market trends, financial news, and economic indicators to identify external risk factors that might impact IT projects.
        1. Gather recent market data and news
        2. Analyze industry-specific trends
        3. Identify economic factors that could affect projects
        4. Assess regulatory changes that might introduce compliance risks
        
        Input: Industry focus (optional) and project context
        Output: Analysis of external market factors that could impact project risks
        """,
        expected_output="""
        A JSON object containing:
        1. Market trends analysis
        2. Economic indicators assessment
        3. Regulatory changes overview
        4. Industry-specific risk factors
        5. Recommendations for monitoring external risks
        """,
        agent=agent,
        tools=[
            tools_dict["market_data_tool"],
            tools_dict["fetch_market_news_tool"],
            tools_dict["identify_external_risks_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are a market analyst specializing in identifying external risk factors that could impact IT projects. Focus on trends that project managers might overlook."}
        ]
    )

def create_risk_scoring_task(agent):
    """Create risk scoring task for the Risk Scoring Agent."""
    return Task(
        description="""
        Evaluate and score identified risks for a project based on probability, impact, and interdependencies.
        1. Analyze each identified risk
        2. Assess probability and impact
        3. Consider risk interdependencies
        4. Calculate overall risk scores
        5. Prioritize risks based on scores
        
        Input: Project ID and list of identified risks
        Output: Detailed risk scoring and prioritization
        """,
        expected_output="""
        A JSON object containing:
        1. Scored risks with probability and impact assessments
        2. Risk interdependency analysis
        3. Overall project risk score
        4. Prioritized list of risks requiring attention
        """,
        agent=agent,
        tools=[
            tools_dict["project_info_tool"],
            tools_dict["project_risks_tool"],
            tools_dict["calculate_risk_score_tool"],
            tools_dict["add_risk_tool"],
            tools_dict["update_risk_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are a risk assessment specialist. Your task is to evaluate risks objectively and assign appropriate probability and impact scores."}
        ]
    )

def create_project_status_task(agent):
    """Create project status tracking task for the Project Status Tracking Agent."""
    return Task(
        description="""
        Track and analyze the current status of a project to identify internal risks such as resource issues, schedule delays, or technical challenges.
        1. Analyze project timeline and milestones
        2. Assess resource allocation and availability
        3. Identify schedule delays or potential bottlenecks
        4. Evaluate technical challenges or implementation issues
        
        Input: Project ID
        Output: Analysis of project status and associated internal risks
        """,
        expected_output="""
        A JSON object containing:
        1. Project status assessment
        2. Timeline analysis with identification of delays
        3. Resource availability assessment
        4. Technical challenges overview
        5. Internal risk factors identified
        """,
        agent=agent,
        tools=[
            tools_dict["project_info_tool"],
            tools_dict["project_risks_tool"],
            tools_dict["add_risk_tool"],
            tools_dict["analyze_project_health_tool"],
            tools_dict["analyze_risk_trends_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are a project status tracking specialist. Your role is to identify internal risk factors by analyzing the current project status, timelines, resources, and technical challenges."}
        ]
    )

def create_reporting_task(agent):
    """Create reporting task for the Reporting Agent."""
    return Task(
        description="""
        Generate comprehensive risk reports for stakeholders and provide actionable recommendations.
        1. Compile risk data from all analyses
        2. Generate a comprehensive risk report
        3. Develop mitigation strategies for high-priority risks
        4. Create visualizations for effective communication
        
        Input: Project ID and compiled risk analyses
        Output: Comprehensive risk report with mitigation strategies
        """,
        expected_output="""
        A JSON object containing:
        1. Executive summary of project risks
        2. Detailed risk analysis by category
        3. Risk visualization data
        4. Mitigation strategies for high-priority risks
        5. Recommendations for ongoing risk monitoring
        """,
        agent=agent,
        tools=[
            tools_dict["project_info_tool"],
            tools_dict["project_risks_tool"],
            tools_dict["generate_risk_report_tool"],
            tools_dict["generate_mitigation_strategies_tool"],
            tools_dict["get_project_reports_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are a risk reporting specialist. Your role is to compile findings from various analyses into a clear, actionable report for stakeholders."}
        ]
    )

def create_mitigation_task(agent):
    """Create mitigation strategies task for the Project Risk Manager agent."""
    return Task(
        description="""
        Develop comprehensive mitigation strategies for identified project risks.
        1. Review high-priority risks
        2. Develop specific mitigation strategies for each risk
        3. Consider resource requirements for mitigation actions
        4. Create implementation plans for mitigation strategies
        
        Input: Project ID and list of prioritized risks
        Output: Detailed mitigation strategies for each high-priority risk
        """,
        expected_output="""
        A JSON object containing:
        1. Mitigation strategies for each high-priority risk
        2. Resource requirements for implementation
        3. Timeline for implementing mitigation actions
        4. Methods for measuring mitigation effectiveness
        """,
        agent=agent,
        tools=[
            tools_dict["project_risks_tool"],
            tools_dict["generate_mitigation_strategies_tool"],
            tools_dict["update_risk_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are developing mitigation strategies for project risks. Focus on practical, actionable steps that can be implemented within the project constraints."}
        ]
    )

def create_risk_monitoring_task(agent):
    """Create risk monitoring task for the Project Risk Manager agent."""
    return Task(
        description="""
        Establish a monitoring framework for ongoing risk assessment and early warning detection.
        1. Define key risk indicators (KRIs) for each major risk
        2. Establish monitoring frequency and methods
        3. Set thresholds for escalation
        4. Create a feedback loop for risk status updates
        
        Input: Project ID and comprehensive risk analysis
        Output: Risk monitoring framework with KRIs and escalation procedures
        """,
        expected_output="""
        A JSON object containing:
        1. Key risk indicators for each major risk category
        2. Monitoring methods and frequency
        3. Escalation thresholds and procedures
        4. Framework for ongoing risk assessment
        """,
        agent=agent,
        tools=[
            tools_dict["project_risks_tool"],
            tools_dict["calculate_risk_score_tool"],
            tools_dict["analyze_risk_trends_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are developing a risk monitoring framework. Focus on early detection of risk escalation through appropriate indicators and thresholds."}
        ]
    )

def create_external_trends_task(agent):
    """Create external trends analysis task for the Market Analysis Agent."""
    return Task(
        description="""
        Analyze long-term market and industry trends that could affect project risks over the project lifecycle.
        1. Identify industry disruption trends
        2. Analyze technological evolution in relevant sectors
        3. Assess long-term economic outlooks
        4. Evaluate geopolitical factors that might impact projects
        
        Input: Project industry and timeline
        Output: Analysis of long-term external trends and their potential impact
        """,
        expected_output="""
        A JSON object containing:
        1. Long-term industry trends analysis
        2. Technological evolution assessment
        3. Economic outlook factors
        4. Geopolitical considerations
        5. Recommendations for strategic risk planning
        """,
        agent=agent,
        tools=[
            tools_dict["market_data_tool"],
            tools_dict["fetch_market_news_tool"],
            tools_dict["identify_external_risks_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are analyzing long-term external trends that could impact project risks over its entire lifecycle. Focus on trends that might not be immediately obvious but could have significant impact later."}
        ]
    )

def create_risk_identification_task(agent):
    """Create risk identification task for the Risk Scoring Agent."""
    return Task(
        description="""
        Identify and categorize potential risks that haven't been captured in the initial project assessment.
        1. Review project details and existing risks
        2. Identify missing or underrepresented risk categories
        3. Generate comprehensive list of additional potential risks
        4. Categorize and provide initial assessment of new risks
        
        Input: Project ID and existing risk analysis
        Output: List of newly identified risks with categorization
        """,
        expected_output="""
        A JSON object containing:
        1. Newly identified risks with descriptions
        2. Categorization of each new risk
        3. Initial probability and impact assessment
        4. Rationale for including each new risk
        """,
        agent=agent,
        tools=[
            tools_dict["project_info_tool"],
            tools_dict["project_risks_tool"],
            tools_dict["add_risk_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are identifying risks that might have been overlooked in initial assessments. Be creative and thorough in considering potential risks across all project dimensions."}
        ]
    )

def create_risk_response_planning_task(agent):
    """Create risk response planning task for the Project Status Tracking Agent."""
    return Task(
        description="""
        Develop detailed response plans for high-impact risks that require immediate action if they occur.
        1. Identify high-impact risks regardless of probability
        2. Develop detailed response plans for each
        3. Define triggers that would activate each response plan
        4. Assign responsibilities for risk response actions
        
        Input: Project ID and prioritized risk list
        Output: Detailed response plans for high-impact risks
        """,
        expected_output="""
        A JSON object containing:
        1. High-impact risks selected for response planning
        2. Detailed response plan for each risk
        3. Trigger conditions for plan activation
        4. Responsibility assignments for response actions
        """,
        agent=agent,
        tools=[
            tools_dict["project_risks_tool"],
            tools_dict["update_risk_tool"],
            tools_dict["generate_mitigation_strategies_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are developing response plans for high-impact risks. Focus on clear, actionable steps that can be taken immediately if the risk materializes."}
        ]
    )

def create_risk_communication_task(agent):
    """Create risk communication task for the Reporting Agent."""
    return Task(
        description="""
        Develop a communication strategy for effectively sharing risk information with different stakeholders.
        1. Identify key stakeholder groups and their information needs
        2. Create tailored risk communication formats for each group
        3. Develop a communication schedule and escalation protocols
        4. Design visualizations that effectively communicate risk status
        
        Input: Project ID and comprehensive risk analysis
        Output: Risk communication strategy for different stakeholders
        """,
        expected_output="""
        A JSON object containing:
        1. Stakeholder analysis with information needs
        2. Communication formats for each stakeholder group
        3. Communication schedule and protocols
        4. Visualization strategies for risk communication
        """,
        agent=agent,
        tools=[
            tools_dict["project_info_tool"],
            tools_dict["project_risks_tool"],
            tools_dict["generate_risk_report_tool"]
        ],
        async_execution=False,
        context=[
            {"role": "system", "content": "You are developing a risk communication strategy. Focus on clear, effective methods for sharing the right level of risk information with different stakeholders."}
        ]
    )

# Create a list of all tasks with their associated agents
def create_all_tasks_with_agents(agents_dict):
    """Create all tasks with their associated agents."""
    tasks = [
        {
            "name": "project_analysis",
            "task": create_project_analysis_task(agents_dict["risk_manager"]),
            "agent": "risk_manager"
        },
        {
            "name": "market_analysis",
            "task": create_market_analysis_task(agents_dict["market_analyst"]),
            "agent": "market_analyst"
        },
        {
            "name": "risk_scoring",
            "task": create_risk_scoring_task(agents_dict["risk_scorer"]),
            "agent": "risk_scorer"
        },
        {
            "name": "project_status",
            "task": create_project_status_task(agents_dict["project_tracker"]),
            "agent": "project_tracker"
        },
        {
            "name": "reporting",
            "task": create_reporting_task(agents_dict["reporting_agent"]),
            "agent": "reporting_agent"
        },
        {
            "name": "mitigation",
            "task": create_mitigation_task(agents_dict["risk_manager"]),
            "agent": "risk_manager"
        },
        {
            "name": "risk_monitoring",
            "task": create_risk_monitoring_task(agents_dict["risk_manager"]),
            "agent": "risk_manager"
        },
        {
            "name": "external_trends",
            "task": create_external_trends_task(agents_dict["market_analyst"]),
            "agent": "market_analyst"
        },
        {
            "name": "risk_identification",
            "task": create_risk_identification_task(agents_dict["risk_scorer"]),
            "agent": "risk_scorer"
        },
        {
            "name": "risk_response_planning",
            "task": create_risk_response_planning_task(agents_dict["project_tracker"]),
            "agent": "project_tracker"
        },
        {
            "name": "risk_communication",
            "task": create_risk_communication_task(agents_dict["reporting_agent"]),
            "agent": "reporting_agent"
        }
    ]
    return tasks
