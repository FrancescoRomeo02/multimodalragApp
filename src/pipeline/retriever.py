from typing import List, Optional, Dict, Any
import logging
from langchain.schema.messages import HumanMessage
import time
import base64

from src.core.models import RetrievalResult
from src.core.prompts import create_prompt_template
from src.utils.qdrant_utils import qdrant_manager
from src.utils.mongodb_utils import mongodb_manager
from src.llm.groq_client import get_groq_llm
from src.utils.performance_monitor import track_performance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _filter_table_context(context_text: str) -> str:
    """
    Filtra il contesto delle tabelle per rimuovere altre tabelle e mantenere solo testo descrittivo utile
    """
    if not context_text:
        return ""
    
    # Dividi il contesto in parti (precedente e successivo)
    parts = context_text.split(" | ")
    filtered_parts = []
    
    for part in parts:
        part_clean = part.strip()
        if not part_clean:
            continue
        
        # Rimuovi etichette come "Contesto precedente:" e "Contesto successivo:"
        if part_clean.startswith("Contesto precedente:"):
            part_clean = part_clean.replace("Contesto precedente:", "").strip()
        elif part_clean.startswith("Contesto successivo:"):
            part_clean = part_clean.replace("Contesto successivo:", "").strip()
        
        # Verifica se il contesto è significativo (non un'altra tabella)
        part_lower = part_clean.lower()
        
        # Indicatori che suggeriscono che il contesto è un'altra tabella
        table_indicators = [
            "|", "---|", "table", "tabella", 
            "row", "column", "cell", "header", "thead", "tbody"
        ]
        
        # Verifica se il contesto contiene principalmente indicatori di tabella
        table_indicator_count = sum(1 for indicator in table_indicators if indicator in part_lower)
        
        # Considera significativo se:
        # 1. È abbastanza lungo (>50 caratteri)
        # 2. Non ha troppi indicatori di tabella (<3)
        # 3. Non è principalmente simboli (|, -, etc.)
        symbol_ratio = sum(1 for char in part_clean if char in "||-") / len(part_clean) if part_clean else 1
        
        if (len(part_clean) > 50 and 
            table_indicator_count < 3 and 
            symbol_ratio < 0.3):  # Meno del 30% di simboli tabella
            
            # Tronca se troppo lungo
            if len(part_clean) > 200:
                part_clean = part_clean[:200] + "..."
            
            filtered_parts.append(part_clean)
    
    return " | ".join(filtered_parts) if filtered_parts else ""


@track_performance(query_type="multimodal_rag")
def enhanced_rag_query(query: str,
                       selected_files: Optional[List[str]] = None) -> RetrievalResult:
    """
    Esegue una query RAG unificata su tutti i tipi di contenuto (testo, immagini, tabelle).
    
    Args:
        query: La domanda da porre al sistema
        selected_files: Lista opzionale di file specifici su cui cercare
    
    Returns:
        RetrievalResult con risposta e documenti recuperati
    """
    logger.info(f"Esecuzione query RAG unificata: '{query}'")
    start_time = time.time()
    
    try:
        # Invece di usare retriever separati, facciamo una ricerca unificata
        query_embedding = qdrant_manager.embedder.embed_query(query)
        
        # Ricerca unificata su tutti i tipi di contenuto
        all_results = qdrant_manager.search_vectors(
            query_embedding=query_embedding,
            selected_files=selected_files or [],
        )
        
        logger.info(f"Trovati {len(all_results)} risultati unificati per query: '{query}'")

        def get_original_document(mongo_id: str) -> Optional[Dict[str, Any]]:
            """Recupera il documento originale da MongoDB usando il riferimento."""
            if not mongo_id:
                return None
            return mongodb_manager.get_document_by_id(mongo_id)
        
        def build_doc_info(result):
            """
            Costruisce le informazioni del documento da un risultato Qdrant,
            arricchite con i dati originali da MongoDB quando disponibili.
            """
            payload = result.payload or {}
            meta = payload.get("metadata", {})
            
            # Estrae il contenuto principale dal page_content
            summary_content = payload.get("page_content", "Contenuto non disponibile")
            
            # Controlla se c'è un riferimento a MongoDB
            mongo_id = meta.get("mongo_id")
            original_doc = get_original_document(mongo_id) if mongo_id else None
            
            # Ottieni il tipo di contenuto originale dai metadati
            original_content_type = "text"  # Valore predefinito
            content = summary_content  # Valore predefinito
            
            # Se c'è il documento MongoDB, usa i dati originali
            if original_doc:
                original_content_type = original_doc.get("content_type", "text")
                
                # Recupera contenuto specifico per tipo
                if original_content_type == "image":
                    content = original_doc.get("caption", summary_content)
                    # Aggiungi riferimento all'immagine originale
                    meta["has_original_image"] = True
                    # Non includiamo l'immagine base64 qui per evitare payload enormi
                
                elif original_content_type == "table":
                    content = original_doc.get("table_markdown", summary_content)
                    meta["has_original_table"] = True
                    # Aggiungi dati strutturati se disponibili
                    if "table_data" in original_doc:
                        meta["has_table_data"] = True
                
                elif original_content_type == "text":
                    content = original_doc.get("content", summary_content)
            
            return {
                "metadata": meta,
                "source": meta.get("source", "Sconosciuto"),
                "page": meta.get("page", "N/A"),
                "content_type": original_content_type,
                "score": result.score,
                "content": content,
                "mongo_id": mongo_id
            }
        
        # Costruisci tutti i documenti recuperati
        source_docs = [build_doc_info(result) for result in all_results]
        
        # Ordina per punteggio di rilevanza (score più alto = più rilevante)
        source_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Log dei tipi di contenuto recuperati per debug
        content_types = {}
        for doc in source_docs:
            doc_type = doc.get("content_type", "unknown")
            content_types[doc_type] = content_types.get(doc_type, 0) + 1
        
        logger.info(f"Contenuti recuperati per tipo: {content_types}")
        
        # Ora passa i documenti al LLM per generare la risposta
        # Costruiamo un contesto unificato con tutti i tipi di contenuto originali
        context_texts = []
        for doc in source_docs[:min(10, len(source_docs))]:  # Limita a 10 documenti per evitare contesti troppo lunghi
            source = doc.get("source", "Sconosciuto")
            page = doc.get("page", "N/A")
            content = doc.get("content", "")
            content_type = doc.get("content_type", "text")
            
            # Formato diverso in base al tipo di contenuto
            if content_type == "image":
                prefix = f"[IMMAGINE da {source}, pagina {page}]\n"
                if doc.get("metadata", {}).get("has_original_image"):
                    prefix += "(Contenuto originale dall'immagine)\n"
            elif content_type == "table":
                prefix = f"[TABELLA da {source}, pagina {page}]\n"
                if doc.get("metadata", {}).get("has_original_table"):
                    prefix += "(Tabella originale con dati completi)\n"
            else:
                prefix = f"[TESTO da {source}, pagina {page}]\n"
                
            context_texts.append(f"{prefix}{content}\n")
        
        unified_context = "\n".join(context_texts)
        
        # Genera la risposta usando il LLM
        llm = get_groq_llm()
        prompt = create_prompt_template()
        
        # Crea una risposta simulando il formato di langchain
        formatted_prompt = prompt.format(context=unified_context, input=query)
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        
        # Estrai correttamente il contenuto della risposta
        if hasattr(response, 'content'):
            answer = str(response.content)
        else:
            answer = str(response)
        
        # Calcola metriche di qualità
        confidence_score = min(1.0, len(source_docs) / 10.0)  # Semplice basato sul numero di risultati
        
        query_time_ms = int((time.time() - start_time) * 1000)
        
        return RetrievalResult(
            answer=answer,
            source_documents=source_docs,
            confidence_score=confidence_score,
            query_time_ms=query_time_ms,
            retrieved_count=len(source_docs),
            filters_applied={
                "selected_files": selected_files, 
                "unified_search": True,
                "content_types_found": content_types
            }
        )

    except Exception as e:
        logger.error(f"Errore RAG unificato: {e}")
        query_time_ms = int((time.time() - start_time) * 1000)
        return RetrievalResult(
            answer="Errore durante la ricerca.",
            source_documents=[],
            confidence_score=0.0,
            query_time_ms=query_time_ms,
            retrieved_count=0,
            filters_applied={"error": str(e)}
        )

