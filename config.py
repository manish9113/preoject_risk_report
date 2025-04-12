import os

# LLM Configuration
LLM_TYPE = "ollama"  # "ollama" instead of "openai"
OLLAMA_MODEL = "llama3"  # or another model you prefer to use
OLLAMA_BASE_URL = "http://localhost:11434"

# Vector Database Configuration
VECTOR_DB_TYPE = "none"  # "chromadb", "pinecone", or "none" to disable
CHROMA_PERSIST_DIRECTORY = "chroma_db"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "project-risks")

# Application Configuration
APP_NAME = "AI Project Risk Management System"
APP_VERSION = "1.0.0"
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Risk Categories and Levels
RISK_CATEGORIES = [
    "Resource",
    "Schedule",
    "Budget",
    "Technical",
    "Quality",
    "Scope",
    "Communication",
    "External",
    "Vendor",
    "Regulatory",
    "Market",
    "Security"
]

RISK_LEVELS = {
    "Low": {"color": "#26eb77", "threshold": 30},
    "Medium": {"color": "#f0cc45", "threshold": 70},
    "High": {"color": "#eb4034", "threshold": 100}
}

# Sample Projects
DEFAULT_PROJECTS = [
    "Cloud Migration",
    "Mobile App Development",
    "ERP Implementation",
    "E-commerce Platform",
    "Data Warehouse Project"
]

# Chat Configuration
MAX_CHAT_HISTORY = 50
CHAT_SAVE_PATH = "chat_history.json"

# Agent System Configuration
AGENT_TEMPERATURE = 0.2
AGENT_PROCESS = "sequential"  # "sequential" or "parallel"

# Data Refresh Configuration
DATA_REFRESH_INTERVAL = 3600  # in seconds (1 hour)
