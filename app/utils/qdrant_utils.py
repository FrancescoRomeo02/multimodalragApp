# app/utils/qdrant_utils.py
from typing import Tuple
import qdrant_client
from qdrant_client.http import models
from app.config import QDRANT_URL, COLLECTION_NAME

def get_qdrant_client():
    """Restituisce un client Qdrant connesso."""
    return qdrant_client.QdrantClient(url=QDRANT_URL)

def delete_documents_by_source(filename: str) -> Tuple[bool, str]:
    """
    Elimina tutti i punti da una collezione Qdrant che hanno 'source' nel payload uguale al nome file.
    """
    print(f"[*] Richiesta di eliminazione per source='{filename}' dalla collezione '{COLLECTION_NAME}'")
    
    client = get_qdrant_client()
    
    # Costruisce il filtro per il campo 'metadata.source' o 'source'
    qdrant_filter = models.Filter(
        should=[
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value=filename),
            ),
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=filename),
            ),
        ],
        must=[],
    )
    
    try:
        response = client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=qdrant_filter),
            wait=True
        )
        print(f"[*] Risultato dell'eliminazione da Qdrant: {response}")
        return True, "Punti eliminati con successo da Qdrant."
    except Exception as e:
        error_message = f"Errore durante l'eliminazione da Qdrant: {e}"
        print(f"[!!!] {error_message}")
        return False, error_message