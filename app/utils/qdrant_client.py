# multimodalrag/app/utils/qdrant_client.py

from qdrant_client import QdrantClient, models
from app.config import QDRANT_URL, COLLECTION_NAME

def get_qdrant_client() -> QdrantClient:
    """
    Restituisce un client Qdrant connesso all'istanza specificata.
    """
    return QdrantClient(url=QDRANT_URL)

def ensure_collection_exists(client: QdrantClient, embedding_dim: int):
    """
    Verifica se la collezione specificata esiste in Qdrant, altrimenti la crea.

    Args:
        client (QdrantClient): Il client Qdrant.
        embedding_dim (int): La dimensione dei vettori che saranno memorizzati.
    """
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"[*] La collezione '{COLLECTION_NAME}' esiste già.")
    except Exception:
        print(f"[*] La collezione '{COLLECTION_NAME}' non esiste. Inizio creazione...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE  # Raccomandato per modelli BGE
            )
        )
        print(f"[*] Collezione '{COLLECTION_NAME}' creata con successo.")