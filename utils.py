import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def setup_cache():
    """Initialize a simple cache for storing temporary data."""
    return {
        "last_update": {},
        "data": {},
    }

# Global cache
cache = setup_cache()

def get_cached_data(key: str, ttl: int = config.CACHE_TTL) -> Optional[Any]:
    """
    Get data from the cache if it exists and is not expired.
    
    Args:
        key: The cache key
        ttl: Time to live in seconds
        
    Returns:
        The cached data or None if not found or expired
    """
    if key not in cache["data"] or key not in cache["last_update"]:
        return None
        
    last_update = cache["last_update"][key]
    if time.time() - last_update > ttl:
        return None
        
    return cache["data"][key]

def set_cached_data(key: str, data: Any) -> None:
    """
    Store data in the cache.
    
    Args:
        key: The cache key
        data: The data to cache
    """
    cache["data"][key] = data
    cache["last_update"][key] = time.time()

def format_timestamp(timestamp: float = None) -> str:
    """
    Format a timestamp as a human-readable string.
    
    Args:
        timestamp: Unix timestamp (default: current time)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def calculate_risk_score(risks: List[Dict]) -> int:
    """
    Calculate an overall risk score based on a list of risks.
    
    Args:
        risks: List of risk dictionaries with 'probability' and 'impact' keys
        
    Returns:
        Overall risk score (0-100)
    """
    if not risks:
        return 0
        
    total_score = 0
    for risk in risks:
        # Convert probability and impact to 0-1 scale if they aren't already
        probability = risk.get("probability", 0)
        if probability > 1:
            probability = probability / 100
            
        impact = risk.get("impact", 0)
        if impact > 1:
            impact = impact / 100
            
        # Calculate individual risk score (0-100)
        risk_score = probability * impact * 100
        
        # Apply category weight if available
        category = risk.get("category", "")
        weight = config.RISK_IMPACT_WEIGHTS.get(category, 1) / 100
        weighted_score = risk_score * weight
        
        total_score += weighted_score
    
    # Normalize the total score to be between 0 and 100
    return min(100, round(total_score))

def get_risk_level(score: int) -> config.RiskLevel:
    """
    Determine the risk level based on the risk score.
    
    Args:
        score: Risk score (0-100)
        
    Returns:
        Risk level enum
    """
    if score <= config.RISK_SCORE_THRESHOLDS[config.RiskLevel.LOW]:
        return config.RiskLevel.LOW
    elif score <= config.RISK_SCORE_THRESHOLDS[config.RiskLevel.MEDIUM]:
        return config.RiskLevel.MEDIUM
    elif score <= config.RISK_SCORE_THRESHOLDS[config.RiskLevel.HIGH]:
        return config.RiskLevel.HIGH
    else:
        return config.RiskLevel.CRITICAL

def create_risk_heatmap(risks: List[Dict]) -> go.Figure:
    """
    Create a risk heatmap visualization.
    
    Args:
        risks: List of risk dictionaries with 'name', 'probability', and 'impact' keys
        
    Returns:
        Plotly figure object
    """
    if not risks:
        return go.Figure()
    
    # Extract data
    names = [risk.get("name", f"Risk {i+1}") for i, risk in enumerate(risks)]
    probabilities = [risk.get("probability", 0) for risk in risks]
    impacts = [risk.get("impact", 0) for risk in risks]
    categories = [risk.get("category", "Unknown") for risk in risks]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        "Risk": names,
        "Probability": probabilities,
        "Impact": impacts,
        "Category": categories
    })
    
    # Create the heatmap
    fig = px.scatter(
        df,
        x="Impact",
        y="Probability",
        text="Risk",
        color="Category",
        size=[1] * len(df),  # Uniform size
        hover_name="Risk",
        title="Project Risk Heatmap",
    )
    
    # Add reference lines for risk levels
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=0.5, y1=0.5,
        fillcolor="green", opacity=0.2, layer="below", line_width=0
    )
    fig.add_shape(
        type="rect",
        x0=0.5, y0=0, x1=1, y1=0.5,
        fillcolor="yellow", opacity=0.2, layer="below", line_width=0
    )
    fig.add_shape(
        type="rect",
        x0=0, y0=0.5, x1=0.5, y1=1,
        fillcolor="yellow", opacity=0.2, layer="below", line_width=0
    )
    fig.add_shape(
        type="rect",
        x0=0.5, y0=0.5, x1=1, y1=1,
        fillcolor="red", opacity=0.2, layer="below", line_width=0
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Impact",
        yaxis_title="Probability",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
    )
    
    return fig

def create_risk_trend_chart(risk_history: List[Dict]) -> go.Figure:
    """
    Create a risk trend chart showing how risk scores have changed over time.
    
    Args:
        risk_history: List of dictionaries with 'date' and 'score' keys
        
    Returns:
        Plotly figure object
    """
    if not risk_history:
        return go.Figure()
    
    # Create DataFrame
    df = pd.DataFrame(risk_history)
    
    # Convert date strings to datetime objects if needed
    if df['date'].dtype == 'object':
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create figure
    fig = px.line(
        df,
        x='date',
        y='score',
        title='Project Risk Score Trend',
        markers=True,
    )
    
    # Add risk level reference lines
    for level in config.RiskLevel:
        threshold = config.RISK_SCORE_THRESHOLDS.get(level, 0)
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"{level.value.capitalize()} Risk",
            annotation_position="bottom right"
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 100]),
    )
    
    return fig

def format_risk_report(project_name: str, risks: List[Dict], overall_score: int) -> str:
    """
    Format a risk report as a markdown string.
    
    Args:
        project_name: Name of the project
        risks: List of risk dictionaries
        overall_score: Overall risk score
        
    Returns:
        Markdown formatted report
    """
    risk_level = get_risk_level(overall_score)
    
    # Start with the header
    report = f"# Risk Report: {project_name}\n\n"
    report += f"**Overall Risk Score:** {overall_score}/100 ({risk_level.value.upper()})\n\n"
    report += f"**Report Generated:** {format_timestamp()}\n\n"
    
    # Add summary of risks by category
    report += "## Risk Summary by Category\n\n"
    
    # Group risks by category
    categories = {}
    for risk in risks:
        category = risk.get("category", "Uncategorized")
        if category not in categories:
            categories[category] = []
        categories[category].append(risk)
    
    # Create a table for each category
    for category, category_risks in categories.items():
        report += f"### {category}\n\n"
        report += "| Risk | Probability | Impact | Score | Status |\n"
        report += "|------|------------|--------|-------|--------|\n"
        
        for risk in category_risks:
            name = risk.get("name", "Unnamed Risk")
            probability = risk.get("probability", 0)
            impact = risk.get("impact", 0)
            score = round(probability * impact * 100)
            status = risk.get("status", "Active")
            
            report += f"| {name} | {probability:.0%} | {impact:.0%} | {score} | {status} |\n"
        
        report += "\n"
    
    # Add mitigation strategies section
    report += "## Mitigation Strategies\n\n"
    
    # List mitigation strategies for high and critical risks
    high_risks = [r for r in risks if r.get("probability", 0) * r.get("impact", 0) > 0.5]
    
    if high_risks:
        for risk in high_risks:
            name = risk.get("name", "Unnamed Risk")
            mitigation = risk.get("mitigation", "No mitigation strategy provided.")
            report += f"### {name}\n\n"
            report += f"{mitigation}\n\n"
    else:
        report += "No high-priority risks requiring immediate mitigation.\n\n"
    
    return report

def json_to_df(json_data: str) -> pd.DataFrame:
    """
    Convert JSON string to pandas DataFrame.
    
    Args:
        json_data: JSON string
        
    Returns:
        Pandas DataFrame
    """
    try:
        data = json.loads(json_data)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            logger.error(f"Unexpected JSON structure: {type(data)}")
            return pd.DataFrame()
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return pd.DataFrame()
