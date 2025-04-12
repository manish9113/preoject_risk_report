import os
import json
import random
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from config import (
    RISK_CATEGORIES, 
    RISK_LEVELS, 
    DEFAULT_PROJECTS,
    OLLAMA_MODEL,
    VECTOR_DB_TYPE,
    CHROMA_PERSIST_DIRECTORY
)

# Mock data generator for development purposes
def generate_mock_risk(project_name: str, date: datetime) -> Dict[str, Any]:
    """Generate a mock risk for development purposes."""
    risk_levels = list(RISK_LEVELS.keys())
    risk_titles = [
        "Resource shortage", "Schedule delay", "Budget overrun", "Technical debt",
        "Quality issues", "Scope creep", "Communication breakdown", "External dependency",
        "Vendor reliability", "Regulatory compliance", "Market shift", "Security vulnerability"
    ]
    
    category = random.choice(RISK_CATEGORIES)
    level = random.choice(risk_levels)
    score = random.randint(
        1, 
        RISK_LEVELS[level]["threshold"]
    )
    
    # Generate a more unique ID using timestamp to avoid duplicates
    unique_id = f"RISK-{int(datetime.now().timestamp() * 1000)}-{random.randint(1000, 9999)}"
    
    return {
        "id": unique_id,
        "title": f"{category} {random.choice(risk_titles)}",
        "description": f"This is a {level.lower()}-level risk related to {category.lower()} for {project_name}.",
        "category": category,
        "level": level,
        "score": score,
        "probability": random.randint(1, 5),
        "impact": random.randint(1, 5),
        "date_identified": date.strftime("%Y-%m-%d"),
        "status": random.choice(["Active", "Mitigated", "Monitoring", "Closed"]),
        "mitigation_strategies": [
            f"Strategy 1 for {category} risk",
            f"Strategy 2 for {category} risk",
            f"Strategy 3 for {category} risk"
        ]
    }

def generate_mock_trend_data(project_name: str, days_back: int) -> List[Dict[str, Any]]:
    """Generate mock trend data for a project."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    trend_data = []
    current_date = start_date
    base_score = random.randint(30, 70)
    
    while current_date <= end_date:
        # Create some random fluctuation in the risk score
        variation = random.randint(-5, 8)
        risk_score = max(10, min(95, base_score + variation))
        
        # Update base score with some momentum
        base_score = base_score + (variation * 0.2)
        
        trend_data.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "risk_score": risk_score,
            "project": project_name
        })
        
        current_date += timedelta(days=1)
    
    return trend_data

def generate_mock_market_data() -> Dict[str, Any]:
    """Generate mock market data for development purposes."""
    return {
        "industry_trends": [
            "Increasing adoption of AI in software development",
            "Remote work becoming standard in IT industry",
            "Growing focus on cybersecurity measures",
            "Shift towards microservices architecture"
        ],
        "economic_indicators": {
            "it_spending_growth": f"{random.uniform(-1.5, 5.5):.1f}%",
            "tech_sector_performance": f"{random.uniform(-2.0, 8.0):.1f}%",
            "industry_confidence_index": random.randint(65, 95)
        },
        "competitor_activities": [
            "Major competitor launched similar product",
            "New startup disrupting traditional market",
            "Industry consolidation through acquisitions"
        ],
        "regulatory_changes": [
            "New data privacy regulations being implemented",
            "Changes in cross-border data transfer rules",
            "Industry-specific compliance requirements updated"
        ],
        "technology_trends": [
            "Rapid advances in machine learning technologies",
            "Increasing cloud adoption across industries",
            "Growing importance of edge computing",
            "Blockchain implementation in enterprise systems"
        ],
        "market_risk_impact": random.choice(["Low", "Medium", "High"])
    }

def generate_mock_risk_by_category(project_name: str) -> List[Dict[str, Any]]:
    """Generate mock risk distribution by category."""
    risk_by_category = []
    
    for category in RISK_CATEGORIES:
        # Not all categories will have risks
        if random.random() > 0.3:  # 70% chance of having risks in this category
            for level in RISK_LEVELS.keys():
                # Not all levels will have risks in each category
                if random.random() > 0.4:  # 60% chance of having risks at this level
                    count = random.randint(1, 5)
                    risk_by_category.append({
                        "category": category,
                        "level": level,
                        "count": count,
                        "project": project_name
                    })
    
    return risk_by_category

def get_project_data(project_name: str, days_back: int = 30) -> Dict[str, Any]:
    """
    Get project data including risks, trends, and metrics.
    
    Args:
        project_name: Name of the project or "All Projects"
        days_back: Number of days of historical data to include
        
    Returns:
        Dictionary containing project data
    """
    # For demo purposes, we'll generate mock data
    # In a real implementation, this would retrieve actual project data from a database
    
    if project_name == "All Projects":
        # Aggregate data from all projects
        all_risks = []
        all_trend_data = []
        all_risk_by_category = []
        
        for project in DEFAULT_PROJECTS:
            # Generate dates from days_back until now
            dates = [datetime.now() - timedelta(days=i) for i in range(days_back)]
            
            # Generate 2-5 risks per project per day (with some randomness)
            for date in dates:
                if random.random() > 0.7:  # 30% chance of generating risks for this day
                    num_risks = random.randint(1, 3)
                    for _ in range(num_risks):
                        all_risks.append(generate_mock_risk(project, date))
            
            # Generate trend data
            all_trend_data.extend(generate_mock_trend_data(project, days_back))
            
            # Generate risk by category
            all_risk_by_category.extend(generate_mock_risk_by_category(project))
        
        # Calculate aggregate metrics
        completion_percentage = random.uniform(0, 100)
        risk_trend = random.uniform(-10, 15)
        mitigation_rate = random.uniform(40, 90)
        
        return {
            "risks": all_risks,
            "trend_data": all_trend_data,
            "risk_by_category": all_risk_by_category,
            "status": "Various",
            "completion_percentage": completion_percentage,
            "risk_trend": risk_trend,
            "mitigation_rate": mitigation_rate,
            "budget_status": "Mixed",
            "resource_utilization": random.uniform(60, 95),
            "start_date": (datetime.now() - timedelta(days=random.randint(90, 180))).strftime("%Y-%m-%d"),
            "end_date": (datetime.now() + timedelta(days=random.randint(30, 180))).strftime("%Y-%m-%d"),
            "key_metrics": {
                "total_projects": len(DEFAULT_PROJECTS),
                "at_risk_projects": random.randint(0, len(DEFAULT_PROJECTS)),
                "on_track_projects": random.randint(0, len(DEFAULT_PROJECTS))
            },
            "market_data": generate_mock_market_data()
        }
    else:
        # Generate data for a specific project
        project_risks = []
        
        # Generate dates from days_back until now
        dates = [datetime.now() - timedelta(days=i) for i in range(days_back)]
        
        # Generate 1-3 risks per day (with some randomness)
        for date in dates:
            if random.random() > 0.7:  # 30% chance of generating risks for this day
                num_risks = random.randint(1, 3)
                for _ in range(num_risks):
                    project_risks.append(generate_mock_risk(project_name, date))
        
        # Generate trend data
        trend_data = generate_mock_trend_data(project_name, days_back)
        
        # Generate risk by category
        risk_by_category = generate_mock_risk_by_category(project_name)
        
        # Calculate project metrics
        completion_percentage = random.uniform(0, 100)
        risk_trend = random.uniform(-10, 15)
        mitigation_rate = random.uniform(40, 90)
        
        return {
            "risks": project_risks,
            "trend_data": trend_data,
            "risk_by_category": risk_by_category,
            "status": random.choice(["On Track", "At Risk", "Delayed", "On Hold"]),
            "completion_percentage": completion_percentage,
            "risk_trend": risk_trend,
            "mitigation_rate": mitigation_rate,
            "budget_status": random.choice(["Under Budget", "On Budget", "Over Budget"]),
            "resource_utilization": random.uniform(60, 95),
            "start_date": (datetime.now() - timedelta(days=random.randint(30, 90))).strftime("%Y-%m-%d"),
            "end_date": (datetime.now() + timedelta(days=random.randint(30, 180))).strftime("%Y-%m-%d"),
            "key_metrics": {
                "tasks_completed": random.randint(10, 100),
                "tasks_remaining": random.randint(0, 50),
                "resource_count": random.randint(5, 20),
                "budget_variance": f"{random.uniform(-15, 15):.1f}%"
            },
            "market_data": generate_mock_market_data()
        }

def load_chat_history() -> List[Dict[str, str]]:
    """
    Load chat history from storage. If no history exists, return a welcome message.
    
    Returns:
        List of chat messages with role and content
    """
    chat_history_path = "chat_history.json"
    
    if os.path.exists(chat_history_path):
        try:
            with open(chat_history_path, "r") as f:
                return json.load(f)
        except Exception:
            # If there's an error loading the file, return default welcome message
            pass
    
    # Default welcome message
    return [
        {
            "role": "assistant", 
            "content": "👋 Welcome to the AI Project Risk Management System! I can help you identify, assess, and mitigate risks across your IT projects. How can I assist you today?"
        }
    ]

def save_chat_history(chat_history: List[Dict[str, str]]) -> None:
    """
    Save chat history to storage.
    
    Args:
        chat_history: List of chat messages with role and content
    """
    chat_history_path = "chat_history.json"
    
    try:
        with open(chat_history_path, "w") as f:
            json.dump(chat_history, f)
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")

def initialize_vector_db():
    """Initialize and return the vector database connection."""
    if VECTOR_DB_TYPE == "chromadb":
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Use in-memory client instead of persistent to avoid tenant issues
            client = chromadb.Client(Settings(is_persistent=False))
            
            # Create collections if they don't exist
            try:
                risks_collection = client.get_or_create_collection("project_risks")
                projects_collection = client.get_or_create_collection("projects")
                market_collection = client.get_or_create_collection("market_data")
                return {
                    "client": client,
                    "collections": {
                        "risks": risks_collection,
                        "projects": projects_collection,
                        "market": market_collection
                    }
                }
            except Exception as e:
                print(f"Error creating ChromaDB collections: {str(e)}")
                return {}
        except ImportError:
            print("ChromaDB not installed. Please install it with 'pip install chromadb'")
            return {}
    elif VECTOR_DB_TYPE == "pinecone":
        try:
            import pinecone
            from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
            
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            
            # Check if index exists, if not this would error in a real implementation
            # but for demo purposes we'll just return the index
            index = pinecone.Index(PINECONE_INDEX_NAME)
            return {
                "client": pinecone,
                "index": index
            }
        except ImportError:
            print("Pinecone not installed. Please install it with 'pip install pinecone-client'")
            return {}
    else:
        print(f"Unsupported vector database type: {VECTOR_DB_TYPE}")
        return {}

def store_risk_data_in_vector_db(risks: List[Dict[str, Any]]) -> bool:
    """
    Store risk data in the vector database.
    
    Args:
        risks: List of risk dictionaries to store
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        vector_db = initialize_vector_db()
        if not vector_db:
            print("Failed to initialize vector database")
            return False
            
        if VECTOR_DB_TYPE == "chromadb":
            collection = vector_db["collections"]["risks"]
            
            # Prepare data for ChromaDB
            ids = [risk["id"] for risk in risks]
            documents = [json.dumps(risk) for risk in risks]
            metadata = [{
                "project": risk.get("project", "Unknown"),
                "category": risk.get("category", "Unknown"),
                "level": risk.get("level", "Unknown"),
                "score": str(risk.get("score", 0))
            } for risk in risks]
            
            # Add data to collection
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadata
            )
            return True
            
        elif VECTOR_DB_TYPE == "pinecone":
            index = vector_db["index"]
            
            # Prepare vectors for Pinecone
            # In a real implementation, you would use an embedding model here
            # For demo purposes, we'll use a simplified approach
            vectors = []
            from langchain_community.embeddings import OllamaEmbeddings
            embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
            
            for risk in risks:
                risk_text = f"{risk.get('title', '')} {risk.get('description', '')}"
                vector = embeddings.embed_query(risk_text)
                vectors.append({
                    "id": risk["id"],
                    "values": vector,
                    "metadata": {
                        "project": risk.get("project", "Unknown"),
                        "category": risk.get("category", "Unknown"),
                        "level": risk.get("level", "Unknown"),
                        "score": risk.get("score", 0),
                        "data": json.dumps(risk)
                    }
                })
            
            # Upsert vectors to Pinecone
            index.upsert(vectors=vectors)
            return True
    except Exception as e:
        print(f"Error storing risk data in vector database: {str(e)}")
        return False

def query_risks_from_vector_db(query: str, project: str = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Query risks from the vector database based on semantic similarity.
    
    Args:
        query: The natural language query
        project: Optional project name to filter by
        limit: Maximum number of results to return
        
    Returns:
        List of risk dictionaries matching the query
    """
    try:
        vector_db = initialize_vector_db()
        if not vector_db:
            print("Failed to initialize vector database")
            return []
            
        if VECTOR_DB_TYPE == "chromadb":
            collection = vector_db["collections"]["risks"]
            
            # Prepare filter if project is specified
            where_filter = {"project": project} if project and project != "All Projects" else None
            
            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter
            )
            
            # Parse results
            risks = []
            if results and "documents" in results and len(results["documents"]) > 0:
                for document in results["documents"][0]:
                    try:
                        risk = json.loads(document)
                        risks.append(risk)
                    except json.JSONDecodeError:
                        continue
            
            return risks
            
        elif VECTOR_DB_TYPE == "pinecone":
            index = vector_db["index"]
            
            # Generate query embedding
            from langchain_community.embeddings import OllamaEmbeddings
            embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
            query_embedding = embeddings.embed_query(query)
            
            # Prepare filter if project is specified
            filter_dict = {"project": {"$eq": project}} if project and project != "All Projects" else None
            
            # Query Pinecone
            query_response = index.query(
                vector=query_embedding,
                top_k=limit,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Parse results
            risks = []
            for match in query_response.matches:
                try:
                    risk = json.loads(match.metadata.get("data", "{}"))
                    risks.append(risk)
                except json.JSONDecodeError:
                    continue
            
            return risks
    except Exception as e:
        print(f"Error querying risks from vector database: {str(e)}")
        return []

def populate_vector_db_with_sample_data():
    """
    Populate the vector database with sample risk data.
    This is for demonstration purposes only.
    """
    # Generate sample risk data for all projects
    all_risks = []
    
    for project in DEFAULT_PROJECTS:
        dates = [datetime.now() - timedelta(days=i) for i in range(30)]
        
        for date in dates:
            num_risks = random.randint(1, 3)
            for _ in range(num_risks):
                risk = generate_mock_risk(project, date)
                # Add project name to risk data
                risk["project"] = project
                all_risks.append(risk)
    
    # Store data in vector DB
    return store_risk_data_in_vector_db(all_risks)
