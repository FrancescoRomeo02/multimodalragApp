# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Percorsi
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "papers_custom_pipeline" # Nome chiaro per la nuova pipeline


# LLM Generativo (usando Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = "llama-3.3-70b-versatile"  # Modello valido disponibile su Groq

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