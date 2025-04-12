from crewai import Task
from typing import List, Optional
import json

# Market Analysis Task
def create_analyze_market_conditions_task(agent, user_query: str, project: str) -> Task:
    """Create a task for analyzing market conditions related to project risks."""
    context = f"The user wants to know about: '{user_query}' for project: '{project}'"
    
    return Task(
        description=f"""
        Analyze market conditions, financial trends, and news that might affect IT project risks.
        
        Focus on:
        1. Industry-specific trends and disruptions
        2. Economic indicators relevant to IT projects
        3. Competitor activities and market movements
        4. Regulatory changes and compliance risks
        5. Technology shifts and obsolescence risks
        
        Context: {context}
        
        Provide a detailed analysis of external market factors that could impact the project(s).
        Format your response as a structured market analysis report with clear sections for 
        different types of external risks.
        """,
        agent=agent,
        expected_output="""
        A comprehensive market analysis report that identifies external risk factors 
        related to the project(s). The report should include:
        - Industry trends affecting the project
        - Economic indicators and their impact
        - Competitive landscape analysis
        - Regulatory and compliance considerations
        - Technology evolution risks
        """,
        context=[{"role": "user", "content": context}]
    )

# Project Status Tracking Task
def create_assess_project_status_task(agent, user_query: str, project: str) -> Task:
    """Create a task for assessing internal project status and risks."""
    context = f"The user wants to know about: '{user_query}' for project: '{project}'"
    
    return Task(
        description=f"""
        Analyze internal project parameters to identify risks related to resources, 
        schedules, and deliverables.
        
        Focus on:
        1. Resource availability and allocation issues
        2. Schedule delays and timeline risks
        3. Budget constraints and financial risks
        4. Quality concerns with deliverables
        5. Team dynamics and communication risks
        
        Context: {context}
        
        Provide a detailed assessment of internal project risks based on the current 
        status of the project(s).
        """,
        agent=agent,
        expected_output="""
        A detailed project status assessment that identifies internal risk factors.
        The assessment should include:
        - Resource-related risks (staffing, skills, availability)
        - Schedule risks and timeline concerns
        - Budget and financial risk factors
        - Quality and deliverable risks
        - Team and communication risks
        """,
        context=[{"role": "user", "content": context}]
    )

# Risk Scoring Task
def create_score_project_risks_task(agent, user_query: str, project: str, dependencies: List[Task]) -> Task:
    """Create a task for scoring and prioritizing identified project risks."""
    context = f"The user wants to know about: '{user_query}' for project: '{project}'"
    
    return Task(
        description=f"""
        Analyze the identified risks from market analysis and project status assessment, 
        then score and prioritize them based on:
        
        1. Probability of occurrence (1-5 scale)
        2. Potential impact severity (1-5 scale)
        3. Overall risk score (calculate as probability × impact)
        4. Urgency (immediate, short-term, long-term)
        5. Controllability (how much the team can mitigate the risk)
        
        Context: {context}
        
        Provide a quantitative assessment of each identified risk with clear scoring 
        and prioritization.
        """,
        agent=agent,
        expected_output="""
        A risk scoring report that quantifies and prioritizes all identified risks.
        The report should include:
        - Individual risk scores (probability, impact, overall score)
        - Risk priority ranking
        - Risk categorization by type and urgency
        - Controllability assessment for each risk
        """,
        context=[{"role": "user", "content": context}],
        dependencies=dependencies
    )

# Risk Assessment Task
def create_generate_risk_assessment_task(agent, user_query: str, project: str, dependencies: List[Task]) -> Task:
    """Create a task for generating comprehensive risk assessment and mitigation strategies."""
    context = f"The user wants to know about: '{user_query}' for project: '{project}'"
    
    return Task(
        description=f"""
        Based on the market analysis, project status assessment, and risk scoring,
        generate a comprehensive risk assessment and develop mitigation strategies:
        
        1. Synthesize insights from all risk analyses
        2. Identify interactions and dependencies between risks
        3. Develop specific mitigation strategies for each high-priority risk
        4. Recommend preventive actions for emerging risks
        5. Suggest contingency plans for unavoidable risks
        
        Context: {context}
        
        Provide a comprehensive assessment that integrates all risk factors and 
        offers actionable mitigation strategies.
        """,
        agent=agent,
        expected_output="""
        A comprehensive risk assessment report with mitigation strategies. The report should include:
        - Integrated risk analysis across all categories
        - Identification of risk interactions and dependencies
        - Specific mitigation strategies for each high-priority risk
        - Preventive actions for emerging risks
        - Contingency plans for unavoidable risks
        - Overall project risk level assessment
        """,
        context=[{"role": "user", "content": context}],
        dependencies=dependencies
    )

# Reporting Task
def create_generate_risk_report_task(agent, user_query: str, project: str, dependencies: List[Task]) -> Task:
    """Create a task for generating the final risk report and recommendations."""
    context = f"The user wants to know about: '{user_query}' for project: '{project}'"
    
    return Task(
        description=f"""
        Based on the comprehensive risk assessment, generate a clear, actionable report 
        that addresses the user's specific query about project risks:
        
        1. Directly answer the user's question about '{user_query}'
        2. Provide relevant risk information specific to the query
        3. Highlight the most critical risks and mitigation strategies
        4. Recommend next steps and actions for decision-makers
        5. Identify any areas requiring further analysis or monitoring
        
        Context: {context}
        
        The report should be conversational and directly address what the user wants to know,
        while providing actionable insights about project risks.
        """,
        agent=agent,
        expected_output="""
        A conversational response that directly addresses the user's query about project risks.
        The response should:
        - Directly answer what the user asked about
        - Provide specific, relevant risk information
        - Highlight critical concerns that need attention
        - Offer clear recommendations for action
        - Be conversational yet informative in tone
        """,
        context=[{"role": "user", "content": context}],
        dependencies=dependencies
    )
