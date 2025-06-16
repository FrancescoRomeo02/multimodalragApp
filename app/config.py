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
EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"
EMBEDDING_DIM = 768

# LLM Generativo (usando Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = "gemma2-9b-it"

# Chunking
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 150