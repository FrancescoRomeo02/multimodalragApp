# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Percorsi
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "papers_custom_pipeline" # Nome chiaro per la nuova pipeline

# Embedding (usando Jina)
JINA_API_KEY = os.getenv("JINA_API_KEY")
EMBEDDING_MODEL_NAME = "llamaindex/vdr-2b-multi-v1"
IMAGE_EMBEDDING_MODEL_NAME = "llamaindex/vdr-2b-multi-v1"
EMBEDDING_DIM = 512

# LLM Generativo (usando Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = "gemma2-9b-it"


# Configurazione processing immagini
UNSTRUCTURED_IMAGE_EXTRACTION_PARAMS = {
    "extract_image_block_types": ["Image", "Figure", "Table"],
    "extract_image_block_to_payload": False,  # Disabilita salvataggio locale
    "extract_image_block_output_dir": None,   # Nessuna directory di output
}

# Aggiungi queste configurazioni
CHUNK_SIZE = 1000  # Caratteri per chunk
CHUNK_OVERLAP = 150  # Caratteri di overlap
MIN_CHUNK_SIZE = 300  # Soglia minima per considerare un chunk valido
