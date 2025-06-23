from typing import List, Optional
from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
import qdrant_client
from langchain.schema.retriever import BaseRetriever
from app.core.models import RetrievalResult
from app.core.diagnostics import validate_retrieval_quality
from app.core.prompts import create_prompt_template
from app.core.qdrant_utils import build_combined_filter, query_images
from app.llm.groq_client import get_groq_llm
from app.config import QDRANT_URL, COLLECTION_NAME
import logging
from app.utils.embedder import get_multimodal_embedding_model

# app/core/rag_service.py
from typing import List
from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain.schema.retriever import BaseRetriever
import qdrant_client


from app.llm.groq_client import get_groq_llm
from app.config import QDRANT_URL, COLLECTION_NAME
from app.utils.embedder import get_multimodal_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_retriever(vector_store: Qdrant, 
                    selected_files: List[str] = None,  # type: ignore
                    query_type: Optional[str] = None,
                    k: int = 5) -> BaseRetriever:
    """
    Crea retriever con filtri appropriati
    """
    qdrant_filter = build_combined_filter(selected_files, query_type)
    
    search_kwargs = {
        "k": k,
        "fetch_k": k * 4
    }
    
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter # type: ignore
    
    return vector_store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance per diversità
        search_kwargs=search_kwargs
    )

def create_rag_chain(selected_files: List[str] = None, multimodal: bool = False): # type: ignore
    """
    Crea catena RAG migliorata con controlli di qualità
    """
    logger.info(f"Creazione catena RAG {'multimodale ' if multimodal else ''}per i file: {selected_files or 'tutti'}")

    # Inizializzazione client e modelli
    try:
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        embedding_model = get_multimodal_embedding_model()
        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embedding_model,
        )
    except Exception as e:
        logger.error(f"Errore inizializzazione Qdrant: {str(e)}")
        raise

    # Configurazione retriever
    retriever = create_retriever(
        vector_store=vector_store, 
        selected_files=selected_files,
        query_type="text" if not multimodal else None,
        k=5
    )
    
    # Prompt template migliorato
    prompt = create_prompt_template()
    
    # LLM configurato
    llm = get_groq_llm()
    
    # Creazione catena RAG
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )
    
    return rag_chain

def enhanced_rag_query(query: str, 
                      selected_files: List[str] = None,  # type: ignore
                      multimodal: bool = False,
                      include_images: bool = False,
                      use_reranking: bool = True,
                      rerank_top_k: int = 5) -> RetrievalResult:
    """
    Query RAG migliorata con validazione qualità e supporto multimodale
    """
    logger.info(f"Esecuzione query RAG: '{query[:50]}...' su file: {selected_files or 'tutti'}")
    
    try:
        # Creazione catena RAG
        rag_chain = create_rag_chain(selected_files, multimodal)
        
        # Esecuzione query principale
        result = rag_chain({"query": query})
        
        # Validazione qualità risultati
        retrieved_docs = result.get("source_documents", [])
        is_quality, confidence_score = validate_retrieval_quality(retrieved_docs)
        
        if not is_quality:
            logger.warning(f"Qualità retrieval insufficiente per query: {query[:50]}...")
        
        # Formattazione documenti fonte
        source_docs = []
        for doc in retrieved_docs:
            doc_info = {
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "Sconosciuto"),
                "page": doc.metadata.get("page", "N/A")
            }
            source_docs.append(doc_info)
        
        # Query immagini se richiesto
        images = []
        if include_images or multimodal:
            try:
                client = qdrant_client.QdrantClient(url=QDRANT_URL)
                images = query_images(client, query, selected_files, top_k=3)
                logger.info(f"Trovate {len(images)} immagini rilevanti")
            except Exception as e:
                logger.warning(f"Errore query immagini: {str(e)}")
        
        return RetrievalResult(
            answer=result["result"],
            source_documents=source_docs,
            images=images if images else None,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        logger.error(f"Errore durante query RAG: {str(e)}")
        return RetrievalResult(
            answer="Si è verificato un errore durante la ricerca. Riprova più tardi.",
            source_documents=[],
            images=None,
            confidence_score=0.0
        )

def batch_query(queries: List[str], 
               selected_files: List[str] = None, # type: ignore
               multimodal: bool = False) -> List[RetrievalResult]:
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

from langchain.schema.messages import HumanMessage

def edit_answer(answer: str):
    llm = get_groq_llm()
    prompt = f"Modifica e migliora la seguente risposta rendendola più chiara e precisa:\n\n{answer}\n\n, comportti come un esperot writer di una rivista di paper scientifici, Risposta migliorata:"
    response = llm([HumanMessage(content=prompt)])
    return response