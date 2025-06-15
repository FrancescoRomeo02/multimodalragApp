import os
from dotenv import load_dotenv

# Carica le variabili dal file .env nella root del progetto
load_dotenv()

# --- Percorsi dei Dati ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")

# --- Configurazione Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# Nome della collezione dedicato a questo modello
COLLECTION_NAME = "papers_jina_v2_base"

# --- Configurazione Modello di Embedding (Solo Jina AI) ---
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"
EMBEDDING_DIM = 768  # Dimensione per jina-v2-base
JINA_API_KEY = os.getenv("JINA_API_KEY")

# --- Configurazione del modello di Visualizzazione ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VLM_MODEL_NAME = "google/vit-base-patch16-224-in21k"  # Modello di visualizzazione predefinito

# --- Parametri di Chunking ---
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 150