from typing import List, Optional
import logging

from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain.schema.retriever import BaseRetriever
from langchain.schema.messages import HumanMessage
import qdrant_client

from app.core.models import RetrievalResult
from app.core.diagnostics import validate_retrieval_quality
from app.core.prompts import create_prompt_template
from app.utils.qdrant_utils import qdrant_manager
from app.utils.embedder import get_multimodal_embedding_model
from app.llm.groq_client import get_groq_llm
from app.config import QDRANT_URL, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_query_type(query: str) -> Optional[str]:
    query_lower = query.lower()
    if any(word in query_lower for word in ["tabella", "table"]):
        return "table"
    if any(word in query_lower for word in ["immagine", "image", "foto"]):
        return "image"
    return "text"

def create_retriever(vector_store: Qdrant, 
                     selected_files: Optional[List[str]] = None,
                     query_type: Optional[str] = None,
                     k: int = 5) -> BaseRetriever:
    qdrant_filter = qdrant_manager.build_combined_filter(
        selected_files=selected_files,
        query_type=query_type
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

def create_rag_chain(query: str,
                     selected_files: Optional[List[str]] = None,
                     multimodal: bool = True):
    logger.info(f"Creazione catena RAG per i file: {selected_files or 'tutti'}")
    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    embedding_model = get_multimodal_embedding_model()
    vector_store = Qdrant(client=client,
                          collection_name=COLLECTION_NAME,
                          embeddings=embedding_model)
    query_type = None if multimodal else infer_query_type(query)

    retriever = create_retriever(vector_store, selected_files, query_type)
    llm = get_groq_llm()
    prompt = create_prompt_template()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"}
    )

def enhanced_rag_query(query: str,
                       selected_files: Optional[List[str]] = None,
                       multimodal: bool = True,
                       include_images: bool = False,
                       include_tables: bool = False) -> RetrievalResult:
    logger.info(f"Esecuzione query RAG: '{query}'")
    try:
        rag_chain = create_rag_chain(query, selected_files, multimodal)
        result = rag_chain({"query": query})
        retrieved_docs = result.get("source_documents", [])
        is_quality, confidence_score = validate_retrieval_quality(retrieved_docs)

        source_docs = []
        for doc in retrieved_docs:
            meta = doc.metadata
            content_type = meta.get("content_type", "text")

            if content_type == "table":
                doc_info = {
                    "content": meta.get("table_markdown", "Contenuto tabella non disponibile"),
                    "metadata": meta,
                    "source": meta.get("source", "Sconosciuto"),
                    "page": meta.get("page", "N/A"),
                    "type": "table",
                    "table_data": meta.get("table_data", {})
                }
            else:
                doc_info = {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": meta,
                    "source": meta.get("source", "Sconosciuto"),
                    "page": meta.get("page", "N/A"),
                    "type": content_type
                }
            source_docs.append(doc_info)

        # Aggiunta immagini se richieste
        images = []
        if include_images or multimodal:
            try:
                images = qdrant_manager.query_images(query, selected_files, top_k=3)
            except Exception as e:
                logger.warning(f"Errore nel recupero immagini: {e}")

        # Aggiunta tabelle se richieste
        tables = []
        if include_tables or multimodal:
            try:
                tables = qdrant_manager.query_tables(query, selected_files, top_k=3)
                for table in tables:
                    source_docs.append({
                        "content": table.get("table_markdown", ""),
                        "metadata": table.get("metadata", {}),
                        "source": table.get("metadata", {}).get("source", "Sconosciuto"),
                        "page": table.get("metadata", {}).get("page", "N/A"),
                        "type": "table",
                        "table_data": table.get("table_data", {})
                    })
            except Exception as e:
                logger.warning(f"Errore nel recupero tabelle: {e}")

        return RetrievalResult(
            answer=result["result"],
            source_documents=source_docs,
            images=images if images else None,
            confidence_score=confidence_score
        )

    except Exception as e:
        logger.error(f"Errore RAG: {e}")
        return RetrievalResult(
            answer="Errore durante la ricerca.",
            source_documents=[],
            confidence_score=0.0
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
                confidence_score=0.0
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
    response = llm([HumanMessage(content=prompt)])
    return response