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
COLLECTION_NAME = "papers_custom_pipeline" # Nome chiaro per la nuova pipeline


# LLM Generativo (usando Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"  # Modello valido disponibile su Groq

# Configurazione processing immagini
UNSTRUCTURED_IMAGE_EXTRACTION_PARAMS = {
    "extract_image_block_types": ["Image", "Figure", "Table"],
    "extract_image_block_to_payload": False,  # Disabilita salvataggio locale
    "extract_image_block_output_dir": None,   # Nessuna directory di output
}

# Aggiungi queste configurazioni
CHUNK_SIZE = 512  # Caratteri per chunk
CHUNK_OVERLAP = 75  # Caratteri di overlap
MIN_CHUNK_SIZE = 200  # Soglia minima per considerare un chunk valido

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_BATCH_SIZE = 32
FALLBACK_TEXT_FOR_EMPTY_DOC = " "
FALLBACK_TEXT_FOR_IMAGE_FAILURE = "Immagine non processabile"

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