import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from config import CHAT_SAVE_PATH, RISK_LEVELS

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history into a string representation."""
    formatted = ""
    for message in chat_history:
        role = message["role"]
        content = message["content"]
        formatted += f"{role.upper()}: {content}\n\n"
    return formatted

def save_chat_history(chat_history: List[Dict[str, str]]) -> None:
    """Save chat history to a file."""
    try:
        with open(CHAT_SAVE_PATH, 'w') as f:
            json.dump(chat_history, f)
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")

def load_chat_history() -> List[Dict[str, str]]:
    """Load chat history from a file if it exists, otherwise return an empty list."""
    try:
        if os.path.exists(CHAT_SAVE_PATH):
            with open(CHAT_SAVE_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading chat history: {str(e)}")
    
    # Return welcome message if chat history doesn't exist or there's an error
    return [
        {
            "role": "assistant",
            "content": """
            👋 Welcome to the AI Project Risk Management System! I can help you identify, 
            assess, and mitigate risks across your IT projects. You can ask me about specific 
            projects, compare risk profiles between projects, or get recommendations for 
            risk mitigation strategies. How can I assist you today?
            """
        }
    ]

def risk_level_from_score(score: float) -> str:
    """Convert a numerical risk score to a risk level string."""
    for level, data in RISK_LEVELS.items():
        if score <= data["threshold"]:
            return level
    return "High"  # Default to high if score exceeds all thresholds

def date_range(days_back: int) -> List[datetime]:
    """Generate a list of dates going back a specified number of days."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    date_list = []
    current_date = start_date
    
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    
    return date_list

def format_timestamp(dt: datetime) -> str:
    """Format a datetime object into a human-readable string."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def generate_risk_report_summary(project_name: str, risks: List[Dict[str, Any]]) -> str:
    """Generate a summary of the risk report for a project."""
    if not risks:
        return "No risks found matching the current filters."
    
    # Count risks by level
    high_risks = len([r for r in risks if r["level"] == "High"])
    medium_risks = len([r for r in risks if r["level"] == "Medium"])
    low_risks = len([r for r in risks if r["level"] == "Low"])
    
    # Get categories with highest risk count
    categories = {}
    for risk in risks:
        category = risk["category"]
        if category in categories:
            categories[category] += 1
        else:
            categories[category] = 1
    
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    top_categories = sorted_categories[:3] if len(sorted_categories) >= 3 else sorted_categories
    
    # Generate summary
    summary = f"## Risk Summary for {project_name}\n\n"
    summary += f"**Total Risks:** {len(risks)}\n\n"
    summary += f"**Risk Breakdown:**\n"
    summary += f"- High: {high_risks}\n"
    summary += f"- Medium: {medium_risks}\n"
    summary += f"- Low: {low_risks}\n\n"
    
    summary += f"**Top Risk Categories:**\n"
    for category, count in top_categories:
        summary += f"- {category}: {count} risks\n"
    
    summary += f"\n**Critical Attention Required:**\n"
    if high_risks > 0:
        critical_risks = [r for r in risks if r["level"] == "High"]
        for i, risk in enumerate(critical_risks[:3]):
            summary += f"{i + 1}. **{risk['title']}** - {risk['description'][:100]}...\n"
    else:
        summary += "No high-level risks identified at this time.\n"
    
    return summary
