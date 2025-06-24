from typing import List, Optional
from qdrant_client.http import models
from typing import List, Optional, Dict, Any
import qdrant_client
from qdrant_client.http import models
from app.config import QDRANT_URL, COLLECTION_NAME
import logging
from app.core.models import ImageResult
from app.utils.embedder import get_multimodal_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_content_filter(query_type: Optional[str] = None) -> Optional[models.Filter]:
    """
    Crea filtri per tipo di contenuto usando i campi STANDARDIZZATI
    """
    if query_type == "image":
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="content_type",
                    match=models.MatchValue(value="image"),
                )
            ]
        )
    elif query_type == "text":
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="text"),
                )
            ]
        )
    return None

def create_file_filter(selected_files: List[str]) -> Optional[models.Filter]:
    """
    Crea filtri per file specifici usando i campi STANDARDIZZATI
    """
    if not selected_files:
        return None
        
    return models.Filter(
        should=[
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value=filename),
            )
            for filename in selected_files
        ]
    )

def create_page_filter(pages: List[int]) -> Optional[models.Filter]:
    """
    Crea filtri per pagine specifiche
    """
    if not pages:
        return None
        
    return models.Filter(
        should=[
            models.FieldCondition(
                key="metadata.page",
                match=models.MatchValue(value=page),
            )
            for page in pages
        ]
    )

def build_combined_filter(selected_files: List[str], 
                         query_type: Optional[str] = None,
                         pages: List[int] = []) -> Optional[models.Filter]:
    """
    Combina filtri multipli
    """
    filters = []
    
    if selected_files:
        file_filter = create_file_filter(selected_files)
        if file_filter:
            filters.append(file_filter)
    
    if query_type:
        content_filter = create_content_filter(query_type)  
        if content_filter:
            filters.append(content_filter)
    
    if pages:
        page_filter = create_page_filter(pages)
        if page_filter:
            filters.append(page_filter)
    
    if not filters:
        return None
    
    if len(filters) == 1:
        return filters[0]
    
    # Combina tutti i filtri con AND
    return models.Filter(must=filters)

def query_images(client: qdrant_client.QdrantClient, 
                query: str, 
                selected_files: List[str] = None,  # type: ignore
                top_k: int = 3) -> List[ImageResult]:
    print(f"--> Esecuzione query immagini: '{query}' con top_k={top_k} e file selezionati: {selected_files}")
    try:
        embedder = get_multimodal_embedding_model()
        query_embedding = embedder.embed_query(query)
        
        qdrant_filter = build_combined_filter(selected_files, "image")
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=None
        )
        
        image_results = []
        print(f"Trovati {len(results)} risultati per la query '{query}'")
        for result in results:
            print(f"Risultato: {result.id}, Punteggio: {result.score}")
            try:
                metadata = result.payload.get("metadata", {}) # type: ignore
                image_base64 = metadata.get("image_base64", "")
                
                if image_base64:
                    image_results.append(ImageResult(
                        image_base64=image_base64,
                        metadata=metadata,
                        score=result.score,
                        page_content=result.payload.get("page_content", "") # type: ignore
                    ))
            except Exception as e:
                logger.warning(f"Errore processamento risultato immagine: {str(e)}")
                continue
        
        return image_results
        
    except Exception as e:
        logger.error(f"Errore durante la query di immagini: {str(e)}")
        return []

def search_similar_documents(query: str, 
                           selected_files: List[str] = None, # type: ignore
                           similarity_threshold: float = 0.7,
                           max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Cerca documenti simili senza elaborazione LLM
    """
    try:
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        embedder = get_multimodal_embedding_model()
        query_embedding = embedder.embed_query(query)
        
        qdrant_filter = build_combined_filter(selected_files, "text")
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=max_results,
            with_payload=True,
            with_vectors=False,
            score_threshold=similarity_threshold
        )
        
        similar_docs = []
        for result in results:
            payload = result.payload or {}
            metadata = payload.get("metadata", {})
            doc_info = {
                "content": payload.get("page_content", ""),
                "metadata": metadata,
                "score": result.score,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "N/A")
            }
            similar_docs.append(doc_info)

        
        return similar_docs
        
    except Exception as e:
        logger.error(f"Errore nella ricerca di documenti simili: {str(e)}")
        return []

def get_document_by_source(source_file: str, page: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        
        # Crea filtro per file specifico
        filters = [
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value=source_file)
            )
        ]
        
        # Aggiungi filtro per pagina se specificato
        if page is not None:
            filters.append(
                models.FieldCondition(
                    key="metadata.page",
                    match=models.MatchValue(value=page)
                )
            )
        
        scroll_filter = models.Filter(must=filters) # type: ignore
        
        # Usa scroll per recuperare tutti i documenti
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=1000,  # Aumenta se necessario
            with_payload=True,
            with_vectors=False
        )
        
        documents = []
        for result in results:
            payload = result.payload or {}
            metadata = payload.get("metadata", {})
            doc_info = {
                "content": payload.get("page_content", ""),
                "metadata": metadata,
                "id": result.id,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "N/A"),
                "type": metadata.get("type", "unknown")
            }
            documents.append(doc_info)
        
        return documents
        
    except Exception as e:
        logger.error(f"Errore nel recupero documenti per fonte {source_file}: {str(e)}")
        return []
