# app/utils/qdrant_utils.py
import qdrant_client
from qdrant_client.http import models
from app.config import QDRANT_URL, COLLECTION_NAME

def get_qdrant_client():
    """Restituisce un client Qdrant connesso."""
    return qdrant_client.QdrantClient(url=QDRANT_URL)

def delete_documents_by_filename(filename: str):
    """
    Elimina tutti i punti da una collezione Qdrant che corrispondono
    a un nome di file specifico nel loro payload/metadata.
    """
    print(f"[*] Richiesta di eliminazione per il file: {filename} dalla collezione '{COLLECTION_NAME}'")
    
    client = get_qdrant_client()
    
    # Crea un filtro per trovare tutti i punti con il filename corrispondente
    qdrant_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.filename", 
                match=models.MatchValue(value=filename),
            )
        ]
    )
    
    # Esegui l'operazione di eliminazione usando il filtro
    try:
        response = client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=qdrant_filter),
            wait=True # Attendi che l'operazione sia completata
        )
        print(f"[*] Risultato dell'eliminazione da Qdrant: {response}")
        return True, "Punti eliminati con successo da Qdrant."
    except Exception as e:
        error_message = f"Errore durante l'eliminazione da Qdrant: {e}"
        print(f"[!!!] {error_message}")
        return False, error_message