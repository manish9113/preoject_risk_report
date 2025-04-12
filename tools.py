from langchain.tools import BaseTool
from typing import List, Dict, Any, Optional
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import random
from pydantic import BaseModel, Field
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from data_handlers import get_project_data
from config import RISK_LEVELS, RISK_CATEGORIES, DEFAULT_PROJECTS

class ProjectDataInput(BaseModel):
    project_name: str = Field(description="The name of the project to get data for, or 'All Projects' for all projects")
    days_back: int = Field(default=30, description="Number of days of historical data to retrieve")

class ProjectInfoTool(BaseTool):
    """Tool for retrieving project information and status."""
    name = "project_info_tool"
    description = """
    Use this tool to get information about project status, including schedule, 
    budget, resources, and other key parameters.
    """
    args_schema = ProjectDataInput
    
    def _run(self, project_name: str, days_back: int = 30, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Get project information."""
        try:
            # Get project data using the data handler
            project_data = get_project_data(project_name, days_back)
            
            # Extract relevant project info
            info = {
                "name": project_name,
                "status": project_data.get("status", "Unknown"),
                "completion_percentage": project_data.get("completion_percentage", 0),
                "budget_status": project_data.get("budget_status", "Unknown"),
                "resource_utilization": project_data.get("resource_utilization", 0),
                "start_date": project_data.get("start_date", "Unknown"),
                "end_date": project_data.get("end_date", "Unknown"),
                "key_metrics": project_data.get("key_metrics", {})
            }
            
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error retrieving project information: {str(e)}"

class RiskAnalysisTool(BaseTool):
    """Tool for analyzing project risks."""
    name = "risk_analysis_tool"
    description = """
    Use this tool to analyze risks for a specific project or across all projects.
    Returns detailed risk information including severity, category, and mitigation strategies.
    """
    args_schema = ProjectDataInput
    
    def _run(self, project_name: str, days_back: int = 30, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Analyze project risks."""
        try:
            # Get project data using the data handler
            project_data = get_project_data(project_name, days_back)
            
            # Extract risk information
            risks = project_data.get("risks", [])
            risk_summary = {
                "project": project_name,
                "total_risks": len(risks),
                "high_priority_risks": len([r for r in risks if r["level"] == "High"]),
                "medium_priority_risks": len([r for r in risks if r["level"] == "Medium"]),
                "low_priority_risks": len([r for r in risks if r["level"] == "Low"]),
                "risk_trend": project_data.get("risk_trend", 0),
                "top_risks": [r for r in risks if r["level"] == "High"][:3],
                "risk_categories": project_data.get("risk_by_category", [])
            }
            
            return json.dumps(risk_summary, indent=2)
        except Exception as e:
            return f"Error analyzing project risks: {str(e)}"

class MarketAnalysisTool(BaseTool):
    """Tool for analyzing market conditions relevant to project risks."""
    name = "market_analysis_tool"
    description = """
    Use this tool to analyze market conditions, industry trends, and external factors
    that might impact project risks.
    """
    args_schema = ProjectDataInput
    
    def _run(self, project_name: str, days_back: int = 30, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Analyze market conditions."""
        try:
            # Get project data using the data handler
            project_data = get_project_data(project_name, days_back)
            
            # Extract market information
            market_info = {
                "industry_trends": project_data.get("market_data", {}).get("industry_trends", []),
                "economic_indicators": project_data.get("market_data", {}).get("economic_indicators", {}),
                "competitor_activities": project_data.get("market_data", {}).get("competitor_activities", []),
                "regulatory_changes": project_data.get("market_data", {}).get("regulatory_changes", []),
                "technology_trends": project_data.get("market_data", {}).get("technology_trends", []),
                "market_risk_impact": project_data.get("market_data", {}).get("market_risk_impact", "Medium")
            }
            
            return json.dumps(market_info, indent=2)
        except Exception as e:
            return f"Error analyzing market conditions: {str(e)}"

class MitigationStrategiesTool(BaseTool):
    """Tool for generating risk mitigation strategies."""
    name = "mitigation_strategies_tool"
    description = """
    Use this tool to generate mitigation strategies for specific risk categories
    or for high-priority risks in a project.
    """
    
    def _run(self, project_name: str, risk_category: Optional[str] = None, risk_level: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Generate mitigation strategies."""
        try:
            # Get project data
            project_data = get_project_data(project_name, 30)
            
            # Filter risks based on category and level if provided
            risks = project_data.get("risks", [])
            if risk_category:
                risks = [r for r in risks if r["category"] == risk_category]
            if risk_level:
                risks = [r for r in risks if r["level"] == risk_level]
            
            # If no risks match the criteria
            if not risks:
                return f"No risks found matching the specified criteria for project '{project_name}'."
            
            # Extract mitigation strategies
            mitigation_info = {
                "project": project_name,
                "risk_count": len(risks),
                "strategies": []
            }
            
            for risk in risks:
                mitigation_info["strategies"].append({
                    "risk_title": risk["title"],
                    "risk_level": risk["level"],
                    "risk_category": risk["category"],
                    "mitigation_strategies": risk["mitigation_strategies"]
                })
            
            return json.dumps(mitigation_info, indent=2)
        except Exception as e:
            return f"Error generating mitigation strategies: {str(e)}"

class ProjectComparisonTool(BaseTool):
    """Tool for comparing risks between multiple projects."""
    name = "project_comparison_tool"
    description = """
    Use this tool to compare risk profiles between different projects.
    Provide a comma-separated list of project names to compare.
    """
    
    def _run(self, projects: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Compare risks between projects."""
        try:
            # Parse project names from comma-separated string
            project_list = [p.strip() for p in projects.split(",")]
            
            # Validate project names
            valid_projects = [p for p in project_list if p in DEFAULT_PROJECTS or p == "All Projects"]
            if not valid_projects:
                return f"No valid projects found in the list: {projects}. Available projects are: {', '.join(DEFAULT_PROJECTS)}"
            
            # Get data for each project
            comparison = {"projects": []}
            
            for project in valid_projects:
                project_data = get_project_data(project, 30)
                
                project_info = {
                    "name": project,
                    "total_risks": len(project_data.get("risks", [])),
                    "high_risks": len([r for r in project_data.get("risks", []) if r["level"] == "High"]),
                    "risk_trend": project_data.get("risk_trend", 0),
                    "top_risk_categories": [c["category"] for c in project_data.get("risk_by_category", [])[:3]],
                    "mitigation_rate": project_data.get("mitigation_rate", 0)
                }
                
                comparison["projects"].append(project_info)
            
            return json.dumps(comparison, indent=2)
        except Exception as e:
            return f"Error comparing projects: {str(e)}"

# Get all available tools
def get_tools() -> List[BaseTool]:
    """Return a list of all available tools."""
    return [
        ProjectInfoTool(),
        RiskAnalysisTool(),
        MarketAnalysisTool(),
        MitigationStrategiesTool(),
        ProjectComparisonTool()
    ]
