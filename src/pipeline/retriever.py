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
    Esegue una query RAG intelligente con rilevamento automatico dell'intent.
    
    Args:
        query: La domanda da porre al sistema
        selected_files: Lista opzionale di file specifici su cui cercare
    
    Returns:
        RetrievalResult con risposta e documenti recuperati
    """
    logger.info(f"Esecuzione query RAG intelligente: '{query}'")
    start_time = time.time()
    
    try:
        # 🧠 UTILIZZO SMART_QUERY - Ricerca intelligente con intent automatico
        search_results = qdrant_manager.smart_query(
            query=query,
            selected_files=selected_files or [],
            content_types=["text", "images", "tables"]
        )
        
        # Estrai metadati sulla query
        query_metadata = search_results.get("query_metadata", {})
        detected_intent = query_metadata.get("intent", "unknown")
        total_results = query_metadata.get("total_results", 0)
        
        logger.info(f"Smart Query completata: intent='{detected_intent}', "
                   f"risultati={total_results} per query: '{query}'")

        # Combina tutti i risultati in una lista unificata per il prompt
        all_results = []
        
        # Aggiungi risultati testo
        for text_result in search_results.get("text", []):
            all_results.append({
                "content": text_result["content"],
                "metadata": text_result["metadata"],
                "score": text_result["score"],
                "content_type": "text",
                "relevance_tier": text_result.get("relevance_tier", "medium")
            })
        
        # Aggiungi risultati immagini  
        for img_result in search_results.get("images", []):
            all_results.append({
                "content": img_result.page_content,
                "metadata": img_result.metadata,
                "score": img_result.score,
                "content_type": "image"
            })
        
        # Aggiungi risultati tabelle
        for table_result in search_results.get("tables", []):
            all_results.append({
                "content": table_result["page_content"],
                "metadata": table_result["metadata"], 
                "score": table_result["score"],
                "content_type": "table",
                "relevance_tier": table_result.get("relevance_tier", "medium")
            })
        
        logger.info(f"Combinati {len(all_results)} risultati totali per generazione risposta")

        # Costruisci documenti dai risultati smart_query
        documents = []
        for result in all_results:
            doc_info = {
                "metadata": result["metadata"],
                "source": result["metadata"].get("source", "Sconosciuto"),
                "page": result["metadata"].get("page", "N/A"),
                "content_type": result["content_type"],
                "score": result["score"],
                "relevance_tier": result.get("relevance_tier", "medium")
            }
            
            if result["content_type"] == "table":
                doc_info.update({
                    "content": result["content"],
                    "caption": result["metadata"].get("caption"),
                    "context_text": result["metadata"].get("context_text", ""),
                    "table_html_raw": result["metadata"].get("table_html")
                })
            elif result["content_type"] == "image":
                doc_info.update({
                    "content": result["content"],
                    "image_base64": result.get("image_base64"),
                    "image_caption": result["metadata"].get("image_caption")
                })
            else:
                # Contenuto testuale
                content = result["content"]
                doc_info["content"] = content[:500] + "..." if len(content) > 500 else content
            
            documents.append(doc_info)
        
        # Ordina per punteggio di rilevanza (score più alto = più rilevante)
        documents.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Log dei tipi di contenuto recuperati per debug  
        content_types = {}
        for doc in documents:
            doc_type = doc.get("content_type", "unknown")
            content_types[doc_type] = content_types.get(doc_type, 0) + 1
        
        logger.info(f"Contenuti recuperati per tipo: {content_types}")
        logger.info(f"Intent rilevato: '{detected_intent}' - Strategia: {query_metadata.get('search_strategy', 'N/A')}")
        
        # Ora passa i documenti al LLM per generare la risposta
        # Costruiamo un contesto unificato con tutti i tipi di contenuto
        context_texts = []
        for doc in documents[:len(documents)]:
            doc_type = doc.get("content_type", "text")
            source = doc.get("source", "Sconosciuto")
            page = doc.get("page", "N/A")
            content = doc.get("content", "")
            relevance_tier = doc.get("relevance_tier", "medium")
            
            if doc_type == "table":
                table_id = doc.get("metadata", {}).get("table_id", "")
                identifier = f"[{table_id}] " if table_id else ""
                context_texts.append(f"[TABELLA {identifier}da {source}, pagina {page}] (Rilevanza: {relevance_tier})\n{content}\n")
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
        confidence_score = min(1.0, len(documents) / 10.0)  # Semplice basato sul numero di risultati
        
        query_time_ms = int((time.time() - start_time) * 1000)
        
        return RetrievalResult(
            answer=answer,
            source_documents=documents,
            confidence_score=confidence_score,
            query_time_ms=query_time_ms,
            retrieved_count=len(documents),
            filters_applied={
                "selected_files": selected_files, 
                "smart_query": True,
                "detected_intent": detected_intent,
                "content_types_found": content_types,
                "search_strategy": query_metadata.get("search_strategy", "N/A")
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

