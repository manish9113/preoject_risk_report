import logging
import time
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests

from langchain.tools import BaseTool, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

from config import (
    OPENAI_API_KEY,
    DEFAULT_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    EXTERNAL_DATA_SOURCES
)
from data_handlers import (
    get_project_by_id,
    get_all_projects,
    get_project_risks,
    get_recent_market_data,
    query_projects,
    query_risks,
    store_risk_data,
    store_project_data,
    store_market_data,
    store_report,
    get_project_reports,
    create_sample_data
)
from utils import (
    calculate_risk_score,
    get_risk_level,
    format_risk_report,
    format_timestamp
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = OpenAI(
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    model_name=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY
)

class ProjectInfoTool(BaseTool):
    name = "project_info_tool"
    description = "Get detailed information about a specific project by ID"
    
    def _run(self, project_id: str) -> Dict:
        """Get project information by ID."""
        logger.info(f"Getting information for project {project_id}")
        project = get_project_by_id(project_id)
        if not project:
            return {"error": f"Project with ID {project_id} not found"}
        return project
    
    def _arun(self, project_id: str) -> Dict:
        """Async version of _run."""
        return self._run(project_id)

class ListProjectsTool(BaseTool):
    name = "list_projects_tool"
    description = "Get a list of all projects"
    
    def _run(self) -> List[Dict]:
        """Get all projects."""
        logger.info("Getting list of all projects")
        projects = get_all_projects()
        return projects
    
    def _arun(self) -> List[Dict]:
        """Async version of _run."""
        return self._run()

class SearchProjectsTool(BaseTool):
    name = "search_projects_tool"
    description = "Search for projects based on query text"
    
    def _run(self, query_text: str) -> List[Dict]:
        """Search projects based on query text."""
        logger.info(f"Searching for projects with query: {query_text}")
        projects = query_projects(query_text)
        return projects
    
    def _arun(self, query_text: str) -> List[Dict]:
        """Async version of _run."""
        return self._run(query_text)

class ProjectRisksTool(BaseTool):
    name = "project_risks_tool"
    description = "Get all risks associated with a specific project"
    
    def _run(self, project_id: str) -> List[Dict]:
        """Get all risks for a project."""
        logger.info(f"Getting risks for project {project_id}")
        risks = get_project_risks(project_id)
        return risks
    
    def _arun(self, project_id: str) -> List[Dict]:
        """Async version of _run."""
        return self._run(project_id)

class CalculateRiskScoreTool(BaseTool):
    name = "calculate_risk_score_tool"
    description = "Calculate the overall risk score for a project based on its risks"
    
    def _run(self, project_id: str) -> Dict:
        """Calculate overall risk score for a project."""
        logger.info(f"Calculating risk score for project {project_id}")
        risks = get_project_risks(project_id)
        if not risks:
            return {
                "project_id": project_id,
                "score": 0,
                "level": "low",
                "risks_count": 0
            }
        
        score = calculate_risk_score(risks)
        level = get_risk_level(score).value
        
        return {
            "project_id": project_id,
            "score": score,
            "level": level,
            "risks_count": len(risks)
        }
    
    def _arun(self, project_id: str) -> Dict:
        """Async version of _run."""
        return self._run(project_id)

class MarketDataTool(BaseTool):
    name = "market_data_tool"
    description = "Get recent market data that might impact project risks"
    
    def _run(self, hours: int = 24, data_type: Optional[str] = None) -> List[Dict]:
        """Get recent market data."""
        logger.info(f"Getting market data from past {hours} hours, type: {data_type}")
        market_data = get_recent_market_data(hours, data_type)
        return market_data
    
    def _arun(self, hours: int = 24, data_type: Optional[str] = None) -> List[Dict]:
        """Async version of _run."""
        return self._run(hours, data_type)

class AddRiskTool(BaseTool):
    name = "add_risk_tool"
    description = "Add a new risk to a project"
    
    def _run(self, risk_data_json: str) -> Dict:
        """Add a new risk."""
        try:
            risk_data = json.loads(risk_data_json)
            logger.info(f"Adding new risk to project {risk_data.get('project_id')}")
            
            # Ensure required fields are present
            required_fields = ["project_id", "name", "category", "probability", "impact"]
            for field in required_fields:
                if field not in risk_data:
                    return {"error": f"Missing required field: {field}"}
            
            # Add timestamp if not present
            if "timestamp" not in risk_data:
                risk_data["timestamp"] = format_timestamp()
            
            # Add id if not present
            if "id" not in risk_data:
                import random
                risk_data["id"] = f"r{random.randint(1000, 9999)}"
            
            # Store the risk
            risk_id = store_risk_data(risk_data)
            return {"success": True, "risk_id": risk_id, "message": "Risk added successfully"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format for risk data"}
        except Exception as e:
            logger.error(f"Error adding risk: {e}")
            return {"error": f"Failed to add risk: {str(e)}"}
    
    def _arun(self, risk_data_json: str) -> Dict:
        """Async version of _run."""
        return self._run(risk_data_json)

class UpdateRiskTool(BaseTool):
    name = "update_risk_tool"
    description = "Update an existing risk"
    
    def _run(self, risk_data_json: str) -> Dict:
        """Update an existing risk."""
        try:
            risk_data = json.loads(risk_data_json)
            logger.info(f"Updating risk {risk_data.get('id')}")
            
            # Ensure id is present
            if "id" not in risk_data:
                return {"error": "Missing required field: id"}
            
            # Store the updated risk
            risk_id = store_risk_data(risk_data)
            return {"success": True, "risk_id": risk_id, "message": "Risk updated successfully"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format for risk data"}
        except Exception as e:
            logger.error(f"Error updating risk: {e}")
            return {"error": f"Failed to update risk: {str(e)}"}
    
    def _arun(self, risk_data_json: str) -> Dict:
        """Async version of _run."""
        return self._run(risk_data_json)

class GenerateRiskReportTool(BaseTool):
    name = "generate_risk_report_tool"
    description = "Generate a comprehensive risk report for a project"
    
    def _run(self, project_id: str) -> Dict:
        """Generate a risk report for a project."""
        logger.info(f"Generating risk report for project {project_id}")
        
        # Get project information
        project = get_project_by_id(project_id)
        if not project:
            return {"error": f"Project with ID {project_id} not found"}
        
        # Get project risks
        risks = get_project_risks(project_id)
        
        # Calculate overall risk score
        overall_score = calculate_risk_score(risks)
        risk_level = get_risk_level(overall_score).value
        
        # Generate report content
        report_content = format_risk_report(project.get("name", "Unnamed Project"), risks, overall_score)
        
        # Prepare summary using LLM
        summary_prompt = PromptTemplate(
            input_variables=["project_name", "risk_level", "risks_count"],
            template="""
            Create a concise summary (1-2 sentences) of the risk situation for project {project_name}.
            The overall risk level is {risk_level} based on {risks_count} identified risks.
            """
        )
        
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        summary = summary_chain.run(
            project_name=project.get("name", "Unnamed Project"),
            risk_level=risk_level,
            risks_count=len(risks)
        )
        
        # Create report data
        report_data = {
            "id": f"report-{project_id}-{int(time.time())}",
            "project_id": project_id,
            "project_name": project.get("name", "Unnamed Project"),
            "timestamp": format_timestamp(),
            "overall_score": overall_score,
            "risk_level": risk_level,
            "risks_count": len(risks),
            "summary": summary.strip(),
            "content": report_content
        }
        
        # Store the report
        report_id = store_report(report_data)
        report_data["report_id"] = report_id
        
        return report_data
    
    def _arun(self, project_id: str) -> Dict:
        """Async version of _run."""
        return self._run(project_id)

class GetProjectReportsTool(BaseTool):
    name = "get_project_reports_tool"
    description = "Get recent risk reports for a project"
    
    def _run(self, project_id: str, limit: int = 5) -> List[Dict]:
        """Get recent risk reports for a project."""
        logger.info(f"Getting reports for project {project_id}")
        reports = get_project_reports(project_id, limit)
        return reports
    
    def _arun(self, project_id: str, limit: int = 5) -> List[Dict]:
        """Async version of _run."""
        return self._run(project_id, limit)

class FetchMarketNewsTool(BaseTool):
    name = "fetch_market_news_tool"
    description = "Fetch the latest market news and trends that might impact project risks"
    
    def _run(self, industry: str = None) -> Dict:
        """Fetch market news and trends."""
        logger.info(f"Fetching market news for industry: {industry}")
        
        # Simulate API call to fetch market news
        # In a real scenario, this would call an actual API
        
        # Generate market news analysis with LLM
        prompt_template = """
        Generate a brief market analysis for the {industry} industry that could impact IT projects.
        Include:
        1. Current market trends
        2. Economic factors
        3. Regulatory changes
        4. Technology shifts
        
        Format the output as a JSON object with the following structure:
        {{
            "trends": [
                {{"title": "Trend title", "description": "Brief description", "impact": "Potential impact on projects"}}
            ],
            "economic_factors": [
                {{"factor": "Factor name", "description": "Brief description", "impact": "Potential impact on projects"}}
            ],
            "regulatory_changes": [
                {{"regulation": "Regulation name", "description": "Brief description", "impact": "Potential impact on projects"}}
            ],
            "technology_shifts": [
                {{"technology": "Technology name", "description": "Brief description", "impact": "Potential impact on projects"}}
            ]
        }}
        """
        
        prompt = PromptTemplate(
            input_variables=["industry"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        industry_value = industry if industry else "technology"
        result = chain.run(industry=industry_value)
        
        try:
            # Parse the generated JSON
            market_data = json.loads(result)
            
            # Store in vector database
            market_entry = {
                "id": f"market-{int(time.time())}",
                "type": "market_analysis",
                "industry": industry_value,
                "timestamp": format_timestamp(),
                "summary": f"Market analysis for {industry_value} industry",
                "details": market_data
            }
            
            store_market_data(market_entry)
            return market_data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM output as JSON: {result}")
            return {
                "error": "Failed to parse market data",
                "raw_output": result
            }
    
    def _arun(self, industry: str = None) -> Dict:
        """Async version of _run."""
        return self._run(industry)

class AnalyzeRiskTrendsTool(BaseTool):
    name = "analyze_risk_trends_tool"
    description = "Analyze trends in project risks over time"
    
    def _run(self, project_id: str) -> Dict:
        """Analyze risk trends for a project."""
        logger.info(f"Analyzing risk trends for project {project_id}")
        
        # Get reports for the project
        reports = get_project_reports(project_id, limit=10)
        
        if not reports:
            return {
                "project_id": project_id,
                "trend": "No historical data available",
                "analysis": "No trend analysis possible without historical data."
            }
        
        # Extract risk scores from reports
        scores = [
            {
                "timestamp": report.get("timestamp", ""),
                "score": report.get("overall_score", 0)
            }
            for report in reports
        ]
        
        # Sort by timestamp
        scores.sort(key=lambda x: x["timestamp"])
        
        # Calculate trend
        if len(scores) >= 2:
            first_score = scores[0]["score"]
            last_score = scores[-1]["score"]
            difference = last_score - first_score
            
            if difference > 5:
                trend = "increasing"
                message = "Risk level is trending upward. Increased attention required."
            elif difference < -5:
                trend = "decreasing"
                message = "Risk level is trending downward. Mitigation strategies may be working."
            else:
                trend = "stable"
                message = "Risk level remains relatively stable."
        else:
            trend = "insufficient_data"
            message = "Not enough historical data to determine a trend."
        
        # Generate analysis with LLM
        if len(scores) >= 2:
            prompt_template = """
            Analyze the trend in risk scores for a project:
            
            Project ID: {project_id}
            Number of reports: {reports_count}
            First score: {first_score} ({first_date})
            Last score: {last_score} ({last_date})
            Overall trend: {trend}
            
            Provide a brief analysis of what this risk trend might indicate and what actions the project team should consider.
            """
            
            prompt = PromptTemplate(
                input_variables=["project_id", "reports_count", "first_score", "first_date", "last_score", "last_date", "trend"],
                template=prompt_template
            )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            analysis = chain.run(
                project_id=project_id,
                reports_count=len(scores),
                first_score=scores[0]["score"],
                first_date=scores[0]["timestamp"],
                last_score=scores[-1]["score"],
                last_date=scores[-1]["timestamp"],
                trend=trend
            )
        else:
            analysis = message
        
        return {
            "project_id": project_id,
            "trend": trend,
            "score_history": scores,
            "analysis": analysis.strip()
        }
    
    def _arun(self, project_id: str) -> Dict:
        """Async version of _run."""
        return self._run(project_id)

class GenerateMitigationStrategiesTool(BaseTool):
    name = "generate_mitigation_strategies_tool"
    description = "Generate mitigation strategies for a specific risk or all project risks"
    
    def _run(self, input_data: str) -> Dict:
        """Generate mitigation strategies."""
        try:
            # Parse the input, which could be a risk ID or a project ID with a flag
            input_json = json.loads(input_data)
            risk_id = input_json.get("risk_id")
            project_id = input_json.get("project_id")
            all_risks = input_json.get("all_risks", False)
            
            if risk_id:
                logger.info(f"Generating mitigation strategy for risk {risk_id}")
                # For a specific risk, query the risk data
                risks = query_risks("", limit=100)  # Get all risks
                target_risk = next((r for r in risks if r.get("id") == risk_id), None)
                
                if not target_risk:
                    return {"error": f"Risk with ID {risk_id} not found"}
                
                # Generate mitigation strategy with LLM
                prompt_template = """
                Generate a comprehensive mitigation strategy for the following risk:
                
                Risk Name: {risk_name}
                Category: {category}
                Description: {description}
                Probability: {probability}
                Impact: {impact}
                
                Your mitigation strategy should include:
                1. Preventive measures
                2. Contingency plans
                3. Monitoring approach
                4. Response procedures
                
                Be specific and actionable in your recommendations.
                """
                
                prompt = PromptTemplate(
                    input_variables=["risk_name", "category", "description", "probability", "impact"],
                    template=prompt_template
                )
                
                chain = LLMChain(llm=llm, prompt=prompt)
                strategy = chain.run(
                    risk_name=target_risk.get("name", "Unnamed Risk"),
                    category=target_risk.get("category", "Uncategorized"),
                    description=target_risk.get("description", "No description"),
                    probability=target_risk.get("probability", 0),
                    impact=target_risk.get("impact", 0)
                )
                
                # Update the risk with the new mitigation strategy
                target_risk["mitigation"] = strategy.strip()
                store_risk_data(target_risk)
                
                return {
                    "risk_id": risk_id,
                    "mitigation_strategy": strategy.strip()
                }
            
            elif project_id and all_risks:
                logger.info(f"Generating mitigation strategies for all risks in project {project_id}")
                # For all project risks
                risks = get_project_risks(project_id)
                
                if not risks:
                    return {"error": f"No risks found for project {project_id}"}
                
                updated_risks = []
                
                for risk in risks:
                    # Generate mitigation strategy with LLM
                    prompt_template = """
                    Generate a concise mitigation strategy for the following risk:
                    
                    Risk Name: {risk_name}
                    Category: {category}
                    Description: {description}
                    Probability: {probability}
                    Impact: {impact}
                    
                    Your mitigation strategy should be specific and actionable.
                    Limit your response to 100 words.
                    """
                    
                    prompt = PromptTemplate(
                        input_variables=["risk_name", "category", "description", "probability", "impact"],
                        template=prompt_template
                    )
                    
                    chain = LLMChain(llm=llm, prompt=prompt)
                    strategy = chain.run(
                        risk_name=risk.get("name", "Unnamed Risk"),
                        category=risk.get("category", "Uncategorized"),
                        description=risk.get("description", "No description"),
                        probability=risk.get("probability", 0),
                        impact=risk.get("impact", 0)
                    )
                    
                    # Update the risk with the new mitigation strategy
                    risk["mitigation"] = strategy.strip()
                    store_risk_data(risk)
                    
                    updated_risks.append({
                        "risk_id": risk.get("id"),
                        "name": risk.get("name"),
                        "mitigation_strategy": strategy.strip()
                    })
                
                return {
                    "project_id": project_id,
                    "updated_risks_count": len(updated_risks),
                    "updated_risks": updated_risks
                }
            
            else:
                return {"error": "Invalid input. Provide either a risk_id or a project_id with all_risks=true"}
        
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}
        except Exception as e:
            logger.error(f"Error generating mitigation strategies: {e}")
            return {"error": f"Failed to generate mitigation strategies: {str(e)}"}
    
    def _arun(self, input_data: str) -> Dict:
        """Async version of _run."""
        return self._run(input_data)

class IdentifyExternalRisksTool(BaseTool):
    name = "identify_external_risks_tool"
    description = "Identify external risk factors that might impact projects based on market data"
    
    def _run(self, project_id: str = None) -> Dict:
        """Identify external risks based on market data."""
        logger.info(f"Identifying external risks for project {project_id}")
        
        # Get recent market data
        market_data = get_recent_market_data(hours=48)
        
        if not market_data:
            return {
                "message": "No recent market data available",
                "external_risks": []
            }
        
        # Get project information if provided
        project = None
        if project_id:
            project = get_project_by_id(project_id)
        
        # Prepare context for LLM
        market_context = "\n".join([
            f"- {data.get('summary', 'No summary')} ({data.get('type', 'Unknown type')})" 
            for data in market_data
        ])
        
        project_context = ""
        if project:
            project_context = f"""
            Project Name: {project.get('name', 'Unnamed project')}
            Description: {project.get('description', 'No description')}
            Industry: {project.get('industry', 'Unknown industry')}
            Status: {project.get('status', 'Unknown status')}
            Budget: {project.get('budget', 'Unknown budget')}
            Timeline: {project.get('start_date', 'Unknown')} to {project.get('end_date', 'Unknown')}
            """
        
        # Generate analysis with LLM
        prompt_template = """
        Identify potential external risk factors based on recent market data that might impact IT projects.
        
        Recent market data:
        {market_context}
        
        {project_specific}
        
        Identify at least 3 but no more than 5 potential external risk factors that could impact IT projects.
        
        For each risk factor:
        1. Provide a name
        2. Describe the risk
        3. Explain how it might impact projects
        4. Suggest how to monitor this risk
        5. Estimate the potential severity (Low, Medium, High)
        
        Format your response as a JSON object with the following structure:
        {{
            "external_risks": [
                {{
                    "name": "Risk factor name",
                    "description": "Description of the risk",
                    "impact": "How it might impact projects",
                    "monitoring": "How to monitor this risk",
                    "severity": "Low|Medium|High"
                }}
            ]
        }}
        """
        
        project_specific = ""
        if project:
            project_specific = f"""
            Project details:
            {project_context}
            
            Focus your analysis on how these external factors might specifically impact this project.
            """
        
        prompt = PromptTemplate(
            input_variables=["market_context", "project_specific"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(
            market_context=market_context,
            project_specific=project_specific
        )
        
        try:
            # Parse the generated JSON
            external_risks = json.loads(result)
            
            # If we have a project, create risk entries for these external factors
            if project and project_id:
                for idx, risk in enumerate(external_risks.get("external_risks", [])):
                    severity_map = {"Low": 0.3, "Medium": 0.6, "High": 0.9}
                    
                    risk_data = {
                        "id": f"r-ext-{project_id}-{int(time.time())}-{idx}",
                        "project_id": project_id,
                        "name": risk.get("name", "Unnamed External Risk"),
                        "description": risk.get("description", "No description"),
                        "category": "External",
                        "probability": 0.5,  # Default probability
                        "impact": severity_map.get(risk.get("severity", "Medium"), 0.6),
                        "status": "Active",
                        "mitigation": f"Monitoring: {risk.get('monitoring', 'No monitoring strategy')}\n\nImpact: {risk.get('impact', 'Unknown impact')}",
                        "source": "External Risk Analysis",
                        "timestamp": format_timestamp()
                    }
                    
                    store_risk_data(risk_data)
            
            return external_risks
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM output as JSON: {result}")
            return {
                "error": "Failed to parse external risks data",
                "raw_output": result
            }
    
    def _arun(self, project_id: str = None) -> Dict:
        """Async version of _run."""
        return self._run(project_id)

class CreateSampleDataTool(BaseTool):
    name = "create_sample_data_tool"
    description = "Create sample project data for demonstration purposes"
    
    def _run(self) -> Dict:
        """Create sample data."""
        logger.info("Creating sample data")
        result = create_sample_data()
        return result
    
    def _arun(self) -> Dict:
        """Async version of _run."""
        return self._run()

class AnalyzeProjectHealthTool(BaseTool):
    name = "analyze_project_health_tool"
    description = "Analyze the overall health of a project based on risks and status"
    
    def _run(self, project_id: str) -> Dict:
        """Analyze overall project health."""
        logger.info(f"Analyzing project health for {project_id}")
        
        # Get project information
        project = get_project_by_id(project_id)
        if not project:
            return {"error": f"Project with ID {project_id} not found"}
        
        # Get project risks
        risks = get_project_risks(project_id)
        
        # Calculate risk score
        risk_score = calculate_risk_score(risks)
        risk_level = get_risk_level(risk_score).value
        
        # Generate health analysis with LLM
        prompt_template = """
        Analyze the overall health of the following project based on its details and risks:
        
        Project Name: {project_name}
        Status: {status}
        Timeline: {start_date} to {end_date}
        Budget: {budget}
        Risk Score: {risk_score}/100 ({risk_level})
        Number of Identified Risks: {risks_count}
        
        Based on this information, provide:
        1. An overall health assessment (Green, Yellow, Red)
        2. A brief explanation of the assessment
        3. Key areas of concern
        4. Top recommendations for improving project health
        
        Format your response as a JSON object with the following structure:
        {{
            "health_status": "Green|Yellow|Red",
            "explanation": "Brief explanation of the health assessment",
            "key_concerns": ["Concern 1", "Concern 2", "Concern 3"],
            "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"]
        }}
        """
        
        prompt = PromptTemplate(
            input_variables=["project_name", "status", "start_date", "end_date", "budget", "risk_score", "risk_level", "risks_count"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(
            project_name=project.get("name", "Unnamed Project"),
            status=project.get("status", "Unknown"),
            start_date=project.get("start_date", "Unknown"),
            end_date=project.get("end_date", "Unknown"),
            budget=project.get("budget", "Unknown"),
            risk_score=risk_score,
            risk_level=risk_level,
            risks_count=len(risks)
        )
        
        try:
            # Parse the generated JSON
            health_analysis = json.loads(result)
            
            # Combine with basic project info
            return {
                "project_id": project_id,
                "project_name": project.get("name", "Unnamed Project"),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "health_analysis": health_analysis
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM output as JSON: {result}")
            return {
                "project_id": project_id,
                "project_name": project.get("name", "Unnamed Project"),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "error": "Failed to parse health analysis",
                "raw_output": result
            }
    
    def _arun(self, project_id: str) -> Dict:
        """Async version of _run."""
        return self._run(project_id)

# Create a list of all tools
all_tools = [
    ProjectInfoTool(),
    ListProjectsTool(),
    SearchProjectsTool(),
    ProjectRisksTool(),
    CalculateRiskScoreTool(),
    MarketDataTool(),
    AddRiskTool(),
    UpdateRiskTool(),
    GenerateRiskReportTool(),
    GetProjectReportsTool(),
    FetchMarketNewsTool(),
    AnalyzeRiskTrendsTool(),
    GenerateMitigationStrategiesTool(),
    IdentifyExternalRisksTool(),
    CreateSampleDataTool(),
    AnalyzeProjectHealthTool(),
]

# Create Langchain tool objects
langchain_tools = [
    Tool(
        name=tool.name,
        func=tool._run,
        description=tool.description
    )
    for tool in all_tools
]

# Dictionary to easily access tools by name
tools_dict = {tool.name: tool for tool in all_tools}
