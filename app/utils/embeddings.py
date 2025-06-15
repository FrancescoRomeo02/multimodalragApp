from llama_index.embeddings.jinaai import JinaEmbedding
from ..config import EMBEDDING_MODEL_NAME, JINA_API_KEY

def get_embedding_model():
    """
    Inizializza e restituisce il modello di embedding di Jina AI,
    utilizzando l'integrazione di LlamaIndex.
    """
    if not JINA_API_KEY:
        raise ValueError("La chiave API di Jina non è stata impostata. Controlla il tuo file .env e assicurati che la variabile JINA_API_KEY sia corretta.")
    
    print(f"[*] Utilizzo del modello di embedding di Jina AI: {EMBEDDING_MODEL_NAME}")
    
    # La documentazione di LlamaIndex per JinaEmbedding specifica
    # il parametro 'api_key'. Passiamolo esplicitamente per massima sicurezza.
    return JinaEmbedding(
        model=EMBEDDING_MODEL_NAME, # Nota: il parametro qui si chiama 'model'
        api_key=JINA_API_KEY
    )