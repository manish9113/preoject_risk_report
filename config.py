import os
from enum import Enum

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Vector Database Configuration
# Choose between ChromaDB or Pinecone
USE_PINECONE = False  # Set to True to use Pinecone instead of ChromaDB
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "project-risks")
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# Agent Configuration
DEFAULT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
MARKET_ANALYSIS_MODEL = "gpt-4o"
RISK_SCORING_MODEL = "gpt-4o"
PROJECT_TRACKING_MODEL = "gpt-4o"
REPORTING_MODEL = "gpt-4o"

# LLM Configuration
TEMPERATURE = 0.2
MAX_TOKENS = 1500

# Project Risk Thresholds
class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

RISK_SCORE_THRESHOLDS = {
    RiskLevel.LOW: 25,
    RiskLevel.MEDIUM: 50,
    RiskLevel.HIGH: 75,
    RiskLevel.CRITICAL: 90,
}

# Risk Categories
RISK_CATEGORIES = [
    "Financial",
    "Resource",
    "Schedule",
    "Technical",
    "Operational",
    "Market",
    "Customer",
    "Regulatory",
    "Strategic"
]

# Risk Impact Weights (out of 100)
RISK_IMPACT_WEIGHTS = {
    "Financial": 25,
    "Resource": 20,
    "Schedule": 15,
    "Technical": 15,
    "Operational": 10,
    "Market": 5,
    "Customer": 5,
    "Regulatory": 3,
    "Strategic": 2
}

# External Data Sources
EXTERNAL_DATA_SOURCES = [
    {"name": "Financial News API", "url": "https://api.financialnews.com"},
    {"name": "Market Trends API", "url": "https://api.markettrends.io"},
    {"name": "Economic Indicators API", "url": "https://api.economicindicators.org"}
]

# Refresh Rate (in seconds)
DATA_REFRESH_RATE = 3600  # Refresh external data every hour
RISK_ASSESSMENT_REFRESH_RATE = 86400  # Reassess project risks daily

# Cache TTL (in seconds)
CACHE_TTL = 3600  # Cache data for one hour
