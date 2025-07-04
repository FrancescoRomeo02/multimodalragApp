# src/config.py
import os
from dotenv import load_dotenv
import logging
from typing import Optional

load_dotenv()

# Percorsi
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "collection_multimodal_rag"


# LLM di Groq e relativa API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Modello LLM generale (per query RAG generiche)
LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Modelli per la descrizione immagini (multimodale)
IMG_DESC_MODEL_SM="meta-llama/llama-4-scout-17b-16e-instruct"
IMG_DESC_MODEL_LG="meta-llama/llama-4-maverick-17b-128e-instruct"

# Modelli per il riassunto di tabelle
TABLE_SUMMARY_MODEL_SM="llama-3.1-8b-instant"
TABLE_SUMMARY_MODEL_LG="llama-3.3-70b-versatile"

# Modelli per il riassunto di testo
TEXT_SUMMARY_MODEL_SM="llama-3.1-8b-instant"
TEXT_SUMMARY_MODEL_LG="llama-3.3-70b-versatile"

#  Modelli per la riscrittura di testo
TEXT_REWRITE_MODEL_SM="llama-3.1-8b-instant"
TEXT_REWRITE_MODEL_LG="llama-3.3-70b-versatile"

#EMBEDDING SETTINFS
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_BATCH_SIZE = 32
FALLBACK_TEXT_FOR_EMPTY_DOC = " "
FALLBACK_TEXT_FOR_IMAGE_FAILURE = "Immagine non processabile"

# SEMANTIC CHUNKING SETTINGS
SEMANTIC_CHUNKING_ENABLED = True           # Flag globale per abilitare chunking semantico
SEMANTIC_CHUNK_SIZE = 1000
SEMANTIC_CHUNK_OVERLAP = 200
SEMANTIC_THRESHOLD = 0.75
MIN_CHUNK_SIZE = 100
CHUNKING_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

# Configurazione Logging Migliorata
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "multimodal_rag.log")

# Performance e Monitoring
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "false").lower() == "true"

# Validazione configurazione critica
def validate_config() -> Optional[str]:
    """Valida la configurazione e restituisce messaggio di errore se invalida"""
    if not GROQ_API_KEY:
        return "GROQ_API_KEY è richiesta ma non configurata"
    
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        try:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        except Exception as e:
            return f"Impossibile creare directory log: {e}"
    
    return None

# Setup logging centralizzato
def setup_logging() -> None:
    """Configura il logging dell'applicazione"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    # Imposta livelli specifici per librerie esterne
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.INFO)