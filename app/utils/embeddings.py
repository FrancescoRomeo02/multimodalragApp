# app/utils/embeddings.py
from langchain.embeddings import JinaEmbeddings
from app.config import EMBEDDING_MODEL_NAME, JINA_API_KEY
from pydantic import SecretStr

def get_embedding_model():
    """Inizializza e restituisce il modello di embedding di Jina AI per LangChain."""
    if not JINA_API_KEY:
        raise ValueError("La chiave API di Jina non è stata impostata (JINA_API_KEY).")
    
    return JinaEmbeddings(
        session=None,  # Usa None per sessione globale
        model_name=EMBEDDING_MODEL_NAME,
        jina_api_key=SecretStr(JINA_API_KEY),
    )