from typing import List, Optional
import logging
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from langchain_qdrant import Qdrant
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

    # Usa la nuova API: create_stuff_documents_chain invece del vecchio approccio
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Creazione della catena di retrieval
    chain = create_retrieval_chain(retriever, document_chain)
    
    return chain

def enhanced_rag_query(query: str,
                       selected_files: Optional[List[str]] = None,
                       multimodal: bool = True,
                       include_images: bool = False,
                       include_tables: bool = False) -> RetrievalResult:
    logger.info(f"Esecuzione query RAG: '{query}'")
    try:
        rag_chain = create_rag_chain(query, selected_files, multimodal)
        result = rag_chain.invoke({"input": query})
        retrieved_docs = result.get("context", [])
        is_quality, confidence_score = validate_retrieval_quality(retrieved_docs)

        def build_doc_info(doc):
            meta = doc.metadata or {}
            content_type = meta.get("content_type", "text")

            base_info = {
                "metadata": meta,
                "source": meta.get("source", "Sconosciuto"),
                "page": meta.get("page", "N/A"),
                "type": content_type
            }

            if content_type == "table":
                table_content = meta.get("table_markdown", "Contenuto tabella non disponibile")
                # Estrai il markdown originale se disponibile, altrimenti usa quello arricchito
                raw_markdown = doc.page_content if hasattr(doc, 'page_content') else table_content
                
                base_info.update({
                    "content": table_content,
                    "table_data": meta.get("table_data", {}),
                    "caption": meta.get("caption"),
                    "context_text": meta.get("context_text"),
                    "table_markdown_raw": raw_markdown
                })
            elif content_type == "image":
                base_info.update({
                    "content": doc.page_content or "",
                    "image_base64": meta.get("image_base64", None),
                    "manual_caption": meta.get("manual_caption"),
                    "context_text": meta.get("context_text"),
                    "image_caption": meta.get("image_caption")  # Per caption generate automaticamente
                })
            else:
                content = doc.page_content or ""
                base_info["content"] = content[:500] + "..." if len(content) > 500 else content

            return base_info

        source_docs = [build_doc_info(doc) for doc in retrieved_docs]

        images = []
        if include_images or multimodal:
            try:
                images = qdrant_manager.query_images(query, selected_files, top_k=3)
                for img in images:
                    source_docs.append({
                        "content": img.page_content,
                        "metadata": img.metadata or {},
                        "source": img.metadata.get("source", "Sconosciuto") if img.metadata else "Sconosciuto",
                        "page": img.metadata.get("page", "N/A") if img.metadata else "N/A",
                        "type": "image",
                        "image_base64": img.image_base64,
                        "manual_caption": img.metadata.get("manual_caption") if img.metadata else None,
                        "context_text": img.metadata.get("context_text") if img.metadata else None,
                        "image_caption": img.metadata.get("image_caption") if img.metadata else None
                    })
            except Exception as e:
                logger.warning(f"Errore nel recupero immagini: {e}")

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
                        "table_data": table.get("table_data", {}),
                        "caption": table.get("metadata", {}).get("caption"),
                        "context_text": table.get("metadata", {}).get("context_text")
                    })
            except Exception as e:
                logger.warning(f"Errore nel recupero tabelle: {e}")

        return RetrievalResult(
            answer=result.get("answer", ""),
            source_documents=source_docs,
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
    response = llm.invoke([HumanMessage(content=prompt)])
    return response