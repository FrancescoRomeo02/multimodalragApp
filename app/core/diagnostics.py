from typing import List, Dict, Any
import qdrant_client
from qdrant_client.http import models
from app.core.qdrant_utils import build_combined_filter
from app.llm.groq_client import get_groq_llm
from app.config import QDRANT_URL, COLLECTION_NAME
from app.utils.embedder import get_multimodal_embedding_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_retrieval_quality(retrieved_docs: List, min_docs: int = 1, min_score: float = 0.3) -> tuple[bool, float]:
    """
    Valida la qualità del retrieval
    """
    if not retrieved_docs or len(retrieved_docs) < min_docs:
        return False, 0.0
    
    # Calcola score medio (se disponibile)
    scores = []
    for doc in retrieved_docs:
        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
            scores.append(doc.metadata['score'])
    
    avg_score = sum(scores) / len(scores) if scores else 0.5
    is_quality = avg_score >= min_score
    
    return is_quality, avg_score


def get_collection_stats() -> Dict[str, Any]:
    try:
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        collection_info = client.get_collection(COLLECTION_NAME)
        
        # Statistiche di base
        stats = {
            "total_vectors": collection_info.vectors_count,
            "collection_status": collection_info.status,
            "optimizer_status": collection_info.optimizer_status,
            "indexed_vectors": collection_info.indexed_vectors_count
        }
        
        # Conteggio per tipo di contenuto se possibile
        try:
            # Conta documenti di testo
            text_count = client.count(
                collection_name=COLLECTION_NAME,
                count_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="metadata.type",
                        match=models.MatchValue(value="text")
                    )]
                )
            )
            # Conta immagini
            image_count = client.count(
                collection_name=COLLECTION_NAME,
                count_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="metadata.type", 
                        match=models.MatchValue(value="image")
                    )]
                )
            )
            
            stats.update({
                "text_documents": text_count.count,
                "image_documents": image_count.count
            })
            
        except Exception as e:
            logger.warning(f"Impossibile ottenere statistiche dettagliate: {str(e)}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Errore ottenimento statistiche: {str(e)}")
        return {"error": str(e)}

def debug_query_results(query: str, selected_files: List[str] = None, top_k: int = 10): # type: ignore
    try:
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        embedder = get_multimodal_embedding_model()
        query_embedding = embedder.embed_query(query)
        
        qdrant_filter = build_combined_filter(selected_files)
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        print(f"\n=== DEBUG QUERY RESULTS ===")
        print(f"Query: {query}")
        print(f"File selezionati: {selected_files or 'Tutti'}")
        print(f"Risultati trovati: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"\n--- Risultato {i+1} ---")
            print(f"Score: {result.score:.4f}")
            payload = result.payload or {}
            metadata = payload.get('metadata', {}) if isinstance(payload.get('metadata', {}), dict) else {}
            print(f"Tipo: {metadata.get('type', 'N/A')}")
            print(f"Fonte: {metadata.get('source', 'N/A')}")
            print(f"Pagina: {metadata.get('page', 'N/A')}")
            print(f"Contenuto: {payload.get('page_content', '')[:200]}...")
            
    except Exception as e:
        print(f"Errore debug: {str(e)}")

def health_check() -> Dict[str, Any]:
    health_status = {
        "retriever": "unknown",
        "qdrant": "unknown", 
        "embeddings": "unknown",
        "llm": "unknown",
        "errors": []
    }
    
    try:
        # Test connessione Qdrant
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        collections = client.get_collections()
        health_status["qdrant"] = "healthy"
    except Exception as e:
        health_status["qdrant"] = "error"
        health_status["errors"].append(f"Qdrant: {str(e)}")
    
    try:
        # Test embedding model
        embedder = get_multimodal_embedding_model()
        test_embedding = embedder.embed_query("test")
        if test_embedding and len(test_embedding) > 0:
            health_status["embeddings"] = "healthy"
        else:
            health_status["embeddings"] = "error"
            health_status["errors"].append("Embeddings: Invalid response")
    except Exception as e:
        health_status["embeddings"] = "error"
        health_status["errors"].append(f"Embeddings: {str(e)}")
    
    try:
        # Test LLM
        llm = get_groq_llm()
        health_status["llm"] = "healthy"
    except Exception as e:
        health_status["llm"] = "error"
        health_status["errors"].append(f"LLM: {str(e)}")
    
    # Status generale
    if not health_status["errors"]:
        health_status["retriever"] = "healthy"
    else:
        health_status["retriever"] = "degraded" if len(health_status["errors"]) < 3 else "error"
    
    return health_status
