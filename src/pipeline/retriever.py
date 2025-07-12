from typing import List, Optional
import logging
from langchain.schema.messages import HumanMessage
import time

from src.core.models import RetrievalResult
from src.core.prompts import create_prompt_template
from src.utils.qdrant_utils import qdrant_manager
from src.llm.groq_client import get_groq_llm
from src.utils.performance_monitor import track_performance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        def build_doc_info(result):
            """Costruisce le informazioni del documento da un risultato Qdrant"""
            payload = result.payload or {}
            meta = payload.get("metadata", {})
            content_type = meta.get("content_type", "text")

            base_info = {
                "metadata": meta,
                "source": meta.get("source", "Sconosciuto"),
                "page": meta.get("page", "N/A"),
                "content_type": content_type,
                "score": result.score  # Aggiungiamo il punteggio per ordinamento
            }
                
            if content_type == "table":
                table_content = payload.get("page_content", meta.get("table_html", "Contenuto tabella non disponibile"))
                
                # Gestione intelligente del contesto per tabelle
                context_text = meta.get("context_text", "")
                
                base_info.update({
                    "content": table_content,
                    "caption": meta.get("caption"),
                    "context_text": context_text,  # Solo contesto significativo
                    "table_html_raw": meta.get("table_html")
                })
            elif content_type == "image":
                base_info.update({
                    "content": payload.get("page_content", ""),
                    "image_base64": payload.get("image_base64", None),
                    "image_caption": meta.get("image_caption")
                })
            else:
                # Contenuto testuale
                content = payload.get("page_content", "")
                base_info["content"] = content[:500] + "..." if len(content) > 500 else content

            return base_info
        
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
        # Costruiamo un contesto unificato con tutti i tipi di contenuto
        context_texts = []
        for doc in source_docs[:len(source_docs)]:
            doc_type = doc.get("content_type", "text")
            source = doc.get("source", "Sconosciuto")
            page = doc.get("page", "N/A")
            content = doc.get("content", "")
            
            if doc_type == "table":
                table_id = doc.get("metadata", {}).get("table_id", "")
                identifier = f"[{table_id}] " if table_id else ""
                context_texts.append(f"[TABELLA {identifier}da {source}, pagina {page}]\n{content}\n")
            elif doc_type == "image":
                image_id = doc.get("metadata", {}).get("image_id", "")
                identifier = f"[{image_id}] " if image_id else ""
                context_texts.append(f"[IMMAGINE {identifier}da {source}, pagina {page}]\n{content}\n")
            else:
                context_texts.append(f"[TESTO da {source}, pagina {page}]\n{content}\n")
        
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

