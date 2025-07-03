from typing import List, Optional
import logging
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from langchain_qdrant import Qdrant
from langchain.schema.retriever import BaseRetriever
from langchain.schema.messages import HumanMessage
import qdrant_client
import time

from src.core.models import RetrievalResult
from src.core.diagnostics import validate_retrieval_quality
from src.core.prompts import create_prompt_template
from src.utils.qdrant_utils import qdrant_manager
from src.utils.embedder import get_multimodal_embedding_model
from src.llm.groq_client import get_groq_llm
from src.config import QDRANT_URL, COLLECTION_NAME, ENABLE_PERFORMANCE_MONITORING
from src.utils.performance_monitor import performance_monitor, track_performance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_retriever(vector_store: Qdrant, 
                     selected_files: Optional[List[str]] = None,
                     k: int = 5) -> BaseRetriever:
    qdrant_filter = qdrant_manager.build_combined_filter(
        selected_files=selected_files or [],
    )

    search_kwargs = {
        "k": k,
        "score_threshold": 0.3
    }
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

def create_rag_chain(selected_files: Optional[List[str]] = None,):
    logger.info(f"Creazione catena RAG per i file: {selected_files or 'tutti'}")

    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    embedding_model = get_multimodal_embedding_model()
    vector_store = Qdrant(client=client,
                         collection_name=COLLECTION_NAME,
                         embeddings=embedding_model)

    retriever = create_retriever(vector_store, selected_files)

    llm = get_groq_llm()
    prompt = create_prompt_template()

    # Usa la nuova API: create_stuff_documents_chain invece del vecchio approccio
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Creazione della catena di retrieval
    chain = create_retrieval_chain(retriever, document_chain)
    
    return chain

@track_performance(query_type="multimodal_rag")
def enhanced_rag_query(query: str,
                       selected_files: Optional[List[str]] = None,
                       multimodal: bool = True,
                       include_images: bool = True,
                       include_tables: bool = True) -> RetrievalResult:
    logger.info(f"Esecuzione query RAG unificata: '{query}'")
    start_time = time.time()
    
    try:
        # Invece di usare retriever separati, facciamo una ricerca unificata
        query_embedding = qdrant_manager.embedder.embed_query(query)
        
        # Ricerca unificata su tutti i tipi di contenuto senza filtri di tipo
        all_results = qdrant_manager.search_vectors(
            query_embedding=query_embedding,
            top_k=15,  # Aumentiamo per avere più risultati misti
            selected_files=selected_files or [],
            query_type=None,  # IMPORTANTE: Non filtriamo per tipo!
            score_threshold=0.3
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
                "type": content_type,
                "score": result.score  # Aggiungiamo il punteggio per ordinamento
            }

            if content_type == "table":
                table_content = payload.get("page_content", meta.get("table_markdown", "Contenuto tabella non disponibile"))
                
                base_info.update({
                    "content": table_content,
                    "table_data": payload.get("table_data", {}),
                    "caption": meta.get("caption"),
                    "context_text": meta.get("context_text"),
                    "table_markdown_raw": meta.get("table_markdown")
                })
            elif content_type == "image":
                base_info.update({
                    "content": payload.get("page_content", ""),
                    "image_base64": payload.get("image_base64", None),
                    "manual_caption": meta.get("manual_caption"),
                    "context_text": meta.get("context_text"),
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
            doc_type = doc.get("type", "unknown")
            content_types[doc_type] = content_types.get(doc_type, 0) + 1
        
        logger.info(f"Contenuti recuperati per tipo: {content_types}")
        
        # Ora passa i documenti al LLM per generare la risposta
        # Costruiamo un contesto unificato con tutti i tipi di contenuto
        context_texts = []
        for doc in source_docs[:10]:  # Prendiamo i top 10 per non sovraccaricare
            doc_type = doc.get("type", "text")
            source = doc.get("source", "Sconosciuto")
            page = doc.get("page", "N/A")
            content = doc.get("content", "")
            
            if doc_type == "table":
                context_texts.append(f"[TABELLA da {source}, pagina {page}]\n{content}\n")
            elif doc_type == "image":
                context_texts.append(f"[IMMAGINE da {source}, pagina {page}]\n{content}\n")
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

def batch_query(queries: List[str], 
               selected_files: List[str] = None, # type: ignore
               multimodal: bool = True) -> List[RetrievalResult]:
    logger.info(f"Esecuzione batch di {len(queries)} query")

    results = []

    for i, query in enumerate(queries):
        try:
            logger.info(f"Processando query {i+1}/{len(queries)}: {query[:30]}...")
            result = enhanced_rag_query(query, selected_files, multimodal)
            results.append(result)
        except Exception as e:
            logger.error(f"Errore query {i+1}: {str(e)}")
            results.append(RetrievalResult(
                answer=f"Errore nella query {i+1}: {str(e)}",
                source_documents=[],
                confidence_score=0.0,
                query_time_ms=0,
                retrieved_count=0,
                filters_applied={"error": str(e)}
            ))

    return results

def edit_answer(answer: str):
    llm = get_groq_llm()
    prompt = f"""Sei un esperto editor accademico. Il tuo compito è revisionare e migliorare la seguente risposta, rendendola:
    - più chiara, precisa e ben strutturata
    - adatta a una rivista scientifica
    - priva di ripetizioni, ambiguità o frasi colloquiali
    - coerente nel tono e nel registro formale
    - in formato markdown
    - non aggiungere alcuna nota aggiuntiva, limitati a revisionare il testo

    Potresti ricevere un messaggio vuoto, in tal caso scrivi solo che non hai informazioni utili
    Agisci come un editor di articoli peer-reviewed. Mantieni intatti i contenuti tecnici e le eventuali referenze, ma migliora forma, logica e stile espositivo.
TESTO ORIGINALE:
{answer}
RISPOSTA REVISIONATA:
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response