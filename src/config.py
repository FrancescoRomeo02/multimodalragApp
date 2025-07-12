# src/config.py
import os
from dotenv import load_dotenv
import logging
from typing import Optional

load_dotenv()

# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "collection_multimodal_rag"


# Groq API and LLM Models
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# General LLM Model (for generic RAG queries)
LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Models for image description (multimodal)
IMG_DESC_MODEL_LG="meta-llama/llama-4-maverick-17b-128e-instruct"

# Models for table summarization
TABLE_SUMMARY_MODEL_LG="llama-3.3-70b-versatile"

# Models for text summarization
TEXT_SUMMARY_MODEL_LG="llama-3.3-70b-versatile"

# Models for text rewriting
TEXT_REWRITE_MODEL_LG="llama-3.3-70b-versatile"

#EMBEDDING SETTINGS
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_BATCH_SIZE = 32
FALLBACK_TEXT_FOR_EMPTY_DOC = " "


# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "multimodal_rag.log")

# Performance and Monitoring
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "false").lower() == "true"

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