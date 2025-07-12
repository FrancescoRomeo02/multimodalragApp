# src/config.py
import os
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any

load_dotenv()

# ===== DIRECTORY CONFIGURATION =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")

# ===== QDRANT CONFIGURATION =====
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "collection_multimodal_rag"

# ===== GROQ API AND LLM MODELS =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# General LLM Model (for generic RAG queries)
LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Specialized models
IMG_DESC_MODEL_LG = "meta-llama/llama-4-maverick-17b-128e-instruct"
TABLE_SUMMARY_MODEL_LG = "llama-3.3-70b-versatile"
TEXT_SUMMARY_MODEL_LG = "llama-3.3-70b-versatile"
TEXT_REWRITE_MODEL_LG = "llama-3.3-70b-versatile"

# ===== EMBEDDING CONFIGURATION =====
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_BATCH_SIZE = 32
FALLBACK_TEXT_FOR_EMPTY_DOC = " "

# ===== RAG SEARCH PARAMETERS (Ottimizzati per qualità) =====
K_NEAREST_NEIGHBORS = 10  # Risultati candidati base

# Score thresholds ottimizzati per tipo di contenuto
SCORE_THRESHOLD_TEXT = 0.65      # Testo ha semantica più flessibile
SCORE_THRESHOLD_IMAGES = 0.70    # Immagini richiedono match più precisi  
SCORE_THRESHOLD_TABLES = 0.60    # Tabelle spesso hanno match più difficili
SCORE_THRESHOLD_MIXED = 0.65     # Per ricerche miste

# Parametri adattivi
ADAPTIVE_K_MIN = 3               # Minimo risultati da restituire
ADAPTIVE_K_MAX = 15              # Massimo risultati da considerare

# Strategie per diversi tipi di query
RAG_PARAMS: Dict[str, Dict[str, Any]] = {
    "factual": {
        "k": 8,
        "score_threshold": 0.70,
        "description": "Per domande fattuali precise"
    },
    "exploratory": {
        "k": 12,
        "score_threshold": 0.60,
        "description": "Per ricerche esplorative ampie"
    },
    "technical": {
        "k": 6,
        "score_threshold": 0.75,
        "description": "Per query tecniche specifiche"
    },
    "multimodal": {
        "k": 10,
        "score_threshold": 0.65,
        "description": "Per ricerche che includono testo, immagini e tabelle"
    }
}


# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "multimodal_rag.log")

# Performance Settings
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# Function to validate configuration
def validate_config() -> Optional[str]:
    """Validates the configuration settings and returns an error message if any required setting is missing."""
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is required but not configured"

    if not os.path.exists(os.path.dirname(LOG_FILE)):
        try:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        except Exception as e:
            return f"Unable to create log directory: {e}"

    return None

# Setup logging
def setup_logging() -> None:
    """Configures the application logging"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    # Set specific levels for external libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.INFO)