import json
import logging
import os
from typing import Dict, List, Optional, Any, Union

import pandas as pd
from datetime import datetime, timedelta
import chromadb
import openai
import random

from config import (
    CHROMA_PERSIST_DIRECTORY,
    USE_PINECONE,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    OPENAI_API_KEY,
    DEFAULT_MODEL,
)
from utils import json_to_df, get_cached_data, set_cached_data, format_timestamp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize vector database
vector_db = None

def init_vector_db():
    """Initialize the vector database (ChromaDB or Pinecone)."""
    global vector_db
    
    if USE_PINECONE:
        try:
            import pinecone
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            
            # Check if index exists, create if it doesn't
            if PINECONE_INDEX_NAME not in pinecone.list_indexes():
                pinecone.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=1536,  # OpenAI embedding dimension
                    metric="cosine"
                )
            
            vector_db = pinecone.Index(PINECONE_INDEX_NAME)
            logger.info(f"Initialized Pinecone index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            # Fall back to ChromaDB
            vector_db = init_chromadb()
    else:
        vector_db = init_chromadb()
    
    return vector_db

def init_chromadb():
    """Initialize ChromaDB."""
    try:
        # Ensure directory exists
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize client
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        
        # Create or get collections
        collections = {
            "projects": client.get_or_create_collection("projects"),
            "risks": client.get_or_create_collection("risks"),
            "market_data": client.get_or_create_collection("market_data"),
            "reports": client.get_or_create_collection("reports")
        }
        
        logger.info("Initialized ChromaDB successfully")
        return collections
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise

def get_embedding(text: str) -> List[float]:
    """
    Get embedding vector for text using OpenAI's embedding API.
    
    Args:
        text: The text to embed
        
    Returns:
        List of embedding values
    """
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        # Return a zero vector as fallback
        return [0.0] * 1536

def store_project_data(project_data: Dict) -> str:
    """
    Store project data in the vector database.
    
    Args:
        project_data: Project data dictionary
        
    Returns:
        ID of stored document
    """
    if vector_db is None:
        init_vector_db()
    
    project_id = project_data.get("id", str(random.randint(10000, 99999)))
    project_name = project_data.get("name", "Unnamed Project")
    
    # Create a text representation of the project for embedding
    project_text = (
        f"Project {project_name} (ID: {project_id}): "
        f"Status: {project_data.get('status', 'Unknown')}. "
        f"Budget: {project_data.get('budget', 'Unknown')}. "
        f"Timeline: {project_data.get('start_date', 'Unknown')} to {project_data.get('end_date', 'Unknown')}. "
        f"Description: {project_data.get('description', 'No description')}."
    )
    
    # Get embedding
    embedding = get_embedding(project_text)
    
    try:
        if USE_PINECONE:
            # Store in Pinecone
            vector_db.upsert(
                vectors=[(project_id, embedding, project_data)],
                namespace="projects"
            )
        else:
            # Store in ChromaDB
            vector_db["projects"].upsert(
                ids=[project_id],
                embeddings=[embedding],
                metadatas=[project_data],
                documents=[project_text]
            )
        
        logger.info(f"Stored project data for {project_name} (ID: {project_id})")
        return project_id
    except Exception as e:
        logger.error(f"Failed to store project data: {e}")
        raise

def store_risk_data(risk_data: Dict) -> str:
    """
    Store risk data in the vector database.
    
    Args:
        risk_data: Risk data dictionary
        
    Returns:
        ID of stored document
    """
    if vector_db is None:
        init_vector_db()
    
    risk_id = risk_data.get("id", str(random.randint(10000, 99999)))
    project_id = risk_data.get("project_id", "unknown")
    risk_name = risk_data.get("name", "Unnamed Risk")
    
    # Create a text representation of the risk for embedding
    risk_text = (
        f"Risk: {risk_name} (ID: {risk_id}) for Project ID {project_id}. "
        f"Category: {risk_data.get('category', 'Unknown')}. "
        f"Probability: {risk_data.get('probability', 0)}. "
        f"Impact: {risk_data.get('impact', 0)}. "
        f"Description: {risk_data.get('description', 'No description')}. "
        f"Mitigation: {risk_data.get('mitigation', 'No mitigation strategy')}."
    )
    
    # Get embedding
    embedding = get_embedding(risk_text)
    
    try:
        if USE_PINECONE:
            # Store in Pinecone
            vector_db.upsert(
                vectors=[(risk_id, embedding, risk_data)],
                namespace="risks"
            )
        else:
            # Store in ChromaDB
            vector_db["risks"].upsert(
                ids=[risk_id],
                embeddings=[embedding],
                metadatas=[risk_data],
                documents=[risk_text]
            )
        
        logger.info(f"Stored risk data for {risk_name} (ID: {risk_id})")
        return risk_id
    except Exception as e:
        logger.error(f"Failed to store risk data: {e}")
        raise

def store_market_data(market_data: Dict) -> str:
    """
    Store market data in the vector database.
    
    Args:
        market_data: Market data dictionary
        
    Returns:
        ID of stored document
    """
    if vector_db is None:
        init_vector_db()
    
    data_id = market_data.get("id", str(random.randint(10000, 99999)))
    data_type = market_data.get("type", "generic")
    timestamp = market_data.get("timestamp", format_timestamp())
    
    # Create a text representation of the market data for embedding
    market_text = (
        f"Market Data (ID: {data_id}): Type: {data_type}. "
        f"Time: {timestamp}. "
        f"Summary: {market_data.get('summary', 'No summary')}. "
        f"Details: {market_data.get('details', 'No details')}."
    )
    
    # Get embedding
    embedding = get_embedding(market_text)
    
    try:
        if USE_PINECONE:
            # Store in Pinecone
            vector_db.upsert(
                vectors=[(data_id, embedding, market_data)],
                namespace="market_data"
            )
        else:
            # Store in ChromaDB
            vector_db["market_data"].upsert(
                ids=[data_id],
                embeddings=[embedding],
                metadatas=[market_data],
                documents=[market_text]
            )
        
        logger.info(f"Stored market data (ID: {data_id}, Type: {data_type})")
        return data_id
    except Exception as e:
        logger.error(f"Failed to store market data: {e}")
        raise

def store_report(report_data: Dict) -> str:
    """
    Store a risk report in the vector database.
    
    Args:
        report_data: Report data dictionary
        
    Returns:
        ID of stored document
    """
    if vector_db is None:
        init_vector_db()
    
    report_id = report_data.get("id", str(random.randint(10000, 99999)))
    project_id = report_data.get("project_id", "unknown")
    timestamp = report_data.get("timestamp", format_timestamp())
    
    # Create a text representation of the report for embedding
    report_text = (
        f"Risk Report (ID: {report_id}) for Project ID {project_id}. "
        f"Generated: {timestamp}. "
        f"Overall Risk Score: {report_data.get('overall_score', 0)}. "
        f"Summary: {report_data.get('summary', 'No summary')}."
    )
    
    # Include full report content if available
    if "content" in report_data:
        report_text += f"\n\nContent: {report_data['content']}"
    
    # Get embedding
    embedding = get_embedding(report_text)
    
    try:
        if USE_PINECONE:
            # Store in Pinecone
            vector_db.upsert(
                vectors=[(report_id, embedding, report_data)],
                namespace="reports"
            )
        else:
            # Store in ChromaDB
            vector_db["reports"].upsert(
                ids=[report_id],
                embeddings=[embedding],
                metadatas=[report_data],
                documents=[report_text]
            )
        
        logger.info(f"Stored risk report (ID: {report_id}) for Project ID {project_id}")
        return report_id
    except Exception as e:
        logger.error(f"Failed to store report: {e}")
        raise

def query_projects(query_text: str, limit: int = 5) -> List[Dict]:
    """
    Query projects based on semantic similarity.
    
    Args:
        query_text: Query string
        limit: Maximum number of results
        
    Returns:
        List of project dictionaries
    """
    if vector_db is None:
        init_vector_db()
    
    # Get embedding for query
    embedding = get_embedding(query_text)
    
    try:
        if USE_PINECONE:
            # Query Pinecone
            results = vector_db.query(
                vector=embedding,
                namespace="projects",
                top_k=limit,
                include_metadata=True
            )
            projects = [match["metadata"] for match in results["matches"]]
        else:
            # Query ChromaDB
            results = vector_db["projects"].query(
                query_embeddings=[embedding],
                n_results=limit,
                include_metadata=True
            )
            projects = results["metadatas"][0] if results["metadatas"] else []
        
        logger.info(f"Query for '{query_text}' returned {len(projects)} projects")
        return projects
    except Exception as e:
        logger.error(f"Failed to query projects: {e}")
        return []

def query_risks(query_text: str, project_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    Query risks based on semantic similarity and optionally filter by project ID.
    
    Args:
        query_text: Query string
        project_id: Optional project ID to filter risks
        limit: Maximum number of results
        
    Returns:
        List of risk dictionaries
    """
    if vector_db is None:
        init_vector_db()
    
    # Get embedding for query
    embedding = get_embedding(query_text)
    
    try:
        if USE_PINECONE:
            # Query Pinecone
            filter_dict = {"project_id": project_id} if project_id else None
            results = vector_db.query(
                vector=embedding,
                namespace="risks",
                filter=filter_dict,
                top_k=limit,
                include_metadata=True
            )
            risks = [match["metadata"] for match in results["matches"]]
        else:
            # Query ChromaDB
            if project_id:
                results = vector_db["risks"].query(
                    query_embeddings=[embedding],
                    where={"project_id": project_id},
                    n_results=limit,
                    include_metadata=True
                )
            else:
                results = vector_db["risks"].query(
                    query_embeddings=[embedding],
                    n_results=limit,
                    include_metadata=True
                )
            risks = results["metadatas"][0] if results["metadatas"] else []
        
        logger.info(f"Query for risks '{query_text}' returned {len(risks)} risks")
        return risks
    except Exception as e:
        logger.error(f"Failed to query risks: {e}")
        return []

def get_project_by_id(project_id: str) -> Optional[Dict]:
    """
    Get a project by its ID.
    
    Args:
        project_id: Project ID
        
    Returns:
        Project dictionary or None if not found
    """
    if vector_db is None:
        init_vector_db()
    
    try:
        if USE_PINECONE:
            results = vector_db.fetch(ids=[project_id], namespace="projects")
            if results["vectors"]:
                return results["vectors"][project_id]["metadata"]
            return None
        else:
            results = vector_db["projects"].get(ids=[project_id], include_metadata=True)
            if results["metadatas"]:
                return results["metadatas"][0]
            return None
    except Exception as e:
        logger.error(f"Failed to get project by ID: {e}")
        return None

def get_project_risks(project_id: str) -> List[Dict]:
    """
    Get all risks associated with a project.
    
    Args:
        project_id: Project ID
        
    Returns:
        List of risk dictionaries
    """
    if vector_db is None:
        init_vector_db()
    
    try:
        if USE_PINECONE:
            results = vector_db.query(
                vector=[0.0] * 1536,  # Dummy vector
                namespace="risks",
                filter={"project_id": project_id},
                top_k=100,  # Fetch a large number to get all risks
                include_metadata=True
            )
            risks = [match["metadata"] for match in results["matches"]]
        else:
            results = vector_db["risks"].get(
                where={"project_id": project_id},
                include_metadata=True
            )
            risks = results["metadatas"] if results["metadatas"] else []
        
        logger.info(f"Retrieved {len(risks)} risks for project ID {project_id}")
        return risks
    except Exception as e:
        logger.error(f"Failed to get project risks: {e}")
        return []

def get_recent_market_data(hours: int = 24, data_type: Optional[str] = None) -> List[Dict]:
    """
    Get recent market data.
    
    Args:
        hours: Number of hours to look back
        data_type: Optional type of market data to filter
        
    Returns:
        List of market data dictionaries
    """
    if vector_db is None:
        init_vector_db()
    
    # Calculate cutoff timestamp
    cutoff_time = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        if USE_PINECONE:
            # Pinecone doesn't support time-based queries directly,
            # so we'll need to fetch all and filter
            filter_dict = {"type": data_type} if data_type else None
            results = vector_db.query(
                vector=[0.0] * 1536,  # Dummy vector
                namespace="market_data",
                filter=filter_dict,
                top_k=100,  # Fetch a large number to then filter by time
                include_metadata=True
            )
            market_data = [
                match["metadata"] for match in results["matches"]
                if match["metadata"].get("timestamp", "") >= cutoff_time
            ]
        else:
            # For ChromaDB, construct the where clause
            where_clause = {}
            if data_type:
                where_clause["type"] = data_type
            
            # We can't do timestamp comparisons directly in ChromaDB,
            # so we'll get all and filter
            results = vector_db["market_data"].get(
                where=where_clause if where_clause else None,
                include_metadata=True
            )
            
            market_data = [
                metadata for metadata in results["metadatas"]
                if metadata.get("timestamp", "") >= cutoff_time
            ]
        
        logger.info(f"Retrieved {len(market_data)} market data entries from the last {hours} hours")
        return market_data
    except Exception as e:
        logger.error(f"Failed to get recent market data: {e}")
        return []

def get_project_reports(project_id: str, limit: int = 5) -> List[Dict]:
    """
    Get recent risk reports for a project.
    
    Args:
        project_id: Project ID
        limit: Maximum number of reports to return
        
    Returns:
        List of report dictionaries
    """
    if vector_db is None:
        init_vector_db()
    
    try:
        if USE_PINECONE:
            results = vector_db.query(
                vector=[0.0] * 1536,  # Dummy vector
                namespace="reports",
                filter={"project_id": project_id},
                top_k=limit,
                include_metadata=True
            )
            reports = [match["metadata"] for match in results["matches"]]
            
            # Sort by timestamp (most recent first)
            reports.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        else:
            results = vector_db["reports"].get(
                where={"project_id": project_id},
                include_metadata=True
            )
            reports = results["metadatas"] if results["metadatas"] else []
            
            # Sort by timestamp (most recent first)
            reports.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
            
            # Limit the number of reports
            reports = reports[:limit]
        
        logger.info(f"Retrieved {len(reports)} risk reports for project ID {project_id}")
        return reports
    except Exception as e:
        logger.error(f"Failed to get project reports: {e}")
        return []

def get_all_projects() -> List[Dict]:
    """
    Get all projects.
    
    Returns:
        List of project dictionaries
    """
    if vector_db is None:
        init_vector_db()
    
    try:
        if USE_PINECONE:
            # Pinecone doesn't support getting all vectors directly,
            # so we'll use a dummy query with a high limit
            results = vector_db.query(
                vector=[0.0] * 1536,  # Dummy vector
                namespace="projects",
                top_k=1000,  # High limit to get all projects
                include_metadata=True
            )
            projects = [match["metadata"] for match in results["matches"]]
        else:
            results = vector_db["projects"].get(include_metadata=True)
            projects = results["metadatas"] if results["metadatas"] else []
        
        logger.info(f"Retrieved {len(projects)} projects")
        return projects
    except Exception as e:
        logger.error(f"Failed to get all projects: {e}")
        return []

def create_sample_data():
    """Create sample data for demonstration purposes."""
    # Sample projects
    projects = [
        {
            "id": "p1001",
            "name": "Cloud Migration",
            "description": "Migrate on-premises infrastructure to cloud services",
            "status": "In Progress",
            "start_date": "2023-01-15",
            "end_date": "2023-07-30",
            "budget": 500000,
            "team_size": 12,
            "client": "InternaCorp",
            "industry": "Finance"
        },
        {
            "id": "p1002",
            "name": "Mobile Banking App",
            "description": "Develop a new mobile banking application with enhanced security",
            "status": "Planning",
            "start_date": "2023-03-01",
            "end_date": "2023-12-15",
            "budget": 750000,
            "team_size": 8,
            "client": "SecureBank",
            "industry": "Banking"
        },
        {
            "id": "p1003",
            "name": "Data Center Upgrade",
            "description": "Upgrade existing data center infrastructure and improve reliability",
            "status": "In Progress",
            "start_date": "2022-11-10",
            "end_date": "2023-05-30",
            "budget": 1200000,
            "team_size": 15,
            "client": "TechGlobal",
            "industry": "Technology"
        }
    ]
    
    # Sample risks for each project
    risks = [
        # Cloud Migration Risks
        {
            "id": "r2001",
            "project_id": "p1001",
            "name": "Data Security Breach",
            "description": "Potential security vulnerabilities during data migration",
            "category": "Security",
            "probability": 0.3,
            "impact": 0.9,
            "status": "Active",
            "mitigation": "Implement end-to-end encryption and conduct security audits before, during, and after migration."
        },
        {
            "id": "r2002",
            "project_id": "p1001",
            "name": "Budget Overrun",
            "description": "Project expenses exceeding the allocated budget",
            "category": "Financial",
            "probability": 0.6,
            "impact": 0.7,
            "status": "Active",
            "mitigation": "Implement strict cost controls and weekly budget reviews."
        },
        {
            "id": "r2003",
            "project_id": "p1001",
            "name": "Service Disruption",
            "description": "Temporary service unavailability during migration",
            "category": "Operational",
            "probability": 0.8,
            "impact": 0.5,
            "status": "Active",
            "mitigation": "Plan for off-hours migration windows and implement redundant systems."
        },
        
        # Mobile Banking App Risks
        {
            "id": "r2004",
            "project_id": "p1002",
            "name": "Regulatory Compliance Issues",
            "description": "Failure to meet financial regulations for mobile banking",
            "category": "Regulatory",
            "probability": 0.4,
            "impact": 0.9,
            "status": "Active",
            "mitigation": "Engage compliance experts and conduct regular regulatory reviews."
        },
        {
            "id": "r2005",
            "project_id": "p1002",
            "name": "Technical Skill Shortage",
            "description": "Lack of specialized mobile security expertise",
            "category": "Resource",
            "probability": 0.7,
            "impact": 0.6,
            "status": "Active",
            "mitigation": "Allocate budget for hiring contractors or training existing staff."
        },
        
        # Data Center Upgrade Risks
        {
            "id": "r2006",
            "project_id": "p1003",
            "name": "Hardware Delivery Delays",
            "description": "Delayed delivery of critical infrastructure components",
            "category": "Schedule",
            "probability": 0.5,
            "impact": 0.7,
            "status": "Active",
            "mitigation": "Order hardware with buffer time and identify alternative suppliers."
        },
        {
            "id": "r2007",
            "project_id": "p1003",
            "name": "Power System Failure",
            "description": "Inadequate power infrastructure for new equipment",
            "category": "Technical",
            "probability": 0.3,
            "impact": 0.8,
            "status": "Active",
            "mitigation": "Conduct power assessment and upgrade power systems before equipment installation."
        },
        {
            "id": "r2008",
            "project_id": "p1003",
            "name": "Staff Resistance",
            "description": "IT operations staff resistant to new technologies",
            "category": "Operational",
            "probability": 0.6,
            "impact": 0.4,
            "status": "Active",
            "mitigation": "Implement change management plan with training and regular communication."
        }
    ]
    
    # Sample market data
    market_data = [
        {
            "id": "m3001",
            "type": "industry_trend",
            "timestamp": format_timestamp(),
            "summary": "Cloud services pricing decreased by 15% on average",
            "details": "Major cloud providers announced price reductions for enterprise customers. This trend could benefit cloud migration projects by reducing ongoing operational costs.",
            "source": "Cloud Industry Report"
        },
        {
            "id": "m3002",
            "type": "economic_indicator",
            "timestamp": format_timestamp(),
            "summary": "Interest rates increased by 0.5%",
            "details": "Central bank raised interest rates, which may impact project financing costs and capital expenditure decisions for IT projects.",
            "source": "Financial Times"
        },
        {
            "id": "m3003",
            "type": "technology_trend",
            "timestamp": format_timestamp(),
            "summary": "Mobile banking adoption increased by 35% year-over-year",
            "details": "Consumer adoption of mobile banking apps continues to accelerate, expanding the potential market but also increasing security concerns and regulatory scrutiny.",
            "source": "Banking Technology Survey"
        },
        {
            "id": "m3004",
            "type": "security_alert",
            "timestamp": format_timestamp(),
            "summary": "New vulnerability discovered in common cloud security protocol",
            "details": "Security researchers identified a critical vulnerability affecting data encryption during cloud migrations. Patches are being developed but haven't been released yet.",
            "source": "Cybersecurity Alert Network"
        }
    ]
    
    # Initialize vector database
    if vector_db is None:
        init_vector_db()
    
    # Store sample data
    for project in projects:
        store_project_data(project)
    
    for risk in risks:
        store_risk_data(risk)
    
    for data in market_data:
        store_market_data(data)
    
    logger.info("Created sample data successfully")
    return {
        "projects": len(projects),
        "risks": len(risks),
        "market_data": len(market_data)
    }

# Initialize vector database when module is imported
try:
    vector_db = init_vector_db()
except Exception as e:
    logger.error(f"Failed to initialize vector database: {e}")
    vector_db = None
