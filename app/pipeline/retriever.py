# app/pipeline/retriever.py
from typing import List, Optional, Dict, Any
from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import qdrant_client
from langchain.schema.retriever import BaseRetriever
from qdrant_client.http import models
from pydantic import BaseModel
from app.utils.embeddings import get_embedding_model
from app.llm.groq_client import get_groq_llm
from app.config import QDRANT_URL, COLLECTION_NAME
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageResult(BaseModel):
    image_base64: str
    metadata: dict
    score: float
    page_content: str

def create_content_filter(query_type: Optional[str] = None) -> Optional[models.Filter]:
    if query_type == "image":
        return models.FieldCondition(
            key="metadata.type",
            match=models.MatchValue(value="image"),
        )
    elif query_type == "text":
        return models.FieldCondition(
            key="metadata.type",
            match=models.MatchValue(value="text"),
        )
    return None

def create_file_filter(selected_files: List[str]) -> Optional[models.Filter]:
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

def build_combined_filter(selected_files: List[str], query_type: Optional[str] = None) -> Optional[models.Filter]:
    file_filter = create_file_filter(selected_files)
    content_filter = create_content_filter(query_type)
    
    filters = []
    if file_filter:
        filters.append(file_filter)
    if content_filter:
        filters.append(content_filter)
    
    if not filters:
        return None
    
    return models.Filter(must=filters) if len(filters) > 1 else filters[0]

def create_retriever(vector_store: Qdrant, selected_files: List[str], query_type: Optional[str] = None) -> BaseRetriever:
    qdrant_filter = build_combined_filter(selected_files, query_type)
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "filter": qdrant_filter,
            "score_threshold": 0.7
        }
    )

#Query su immagini non funzionante poichè non storate nel db
def query_images(client: qdrant_client.QdrantClient, 
                query: str, 
                selected_files: List[str], 
                top_k: int = 3) -> List[ImageResult]:
    try:
        embedder = get_embedding_model()
        query_embedding = embedder.embed_query(query)
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=build_combined_filter(selected_files, "image"),
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            ImageResult(
                image_base64=result.payload["metadata"].get("image_base64", ""),
                metadata=result.payload["metadata"],
                score=result.score,
                page_content=result.payload.get("page_content", "")
            )
            for result in results
            if "image_base64" in result.payload.get("metadata", {})
        ]
    except Exception as e:
        logger.error(f"Errore durante la query di immagini: {str(e)}")
        return []

def create_rag_chain(selected_files: List[str], multimodal: bool = False):
    logger.info(f"Creazione catena RAG {'multimodale ' if multimodal else ''}per i file: {selected_files}")

    # Inizializzazione client e modelli
    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    embedding_model = get_embedding_model()
    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
    )

    template = """Sei un assistente di ricerca accademico che deve spiegare all'utente argomenti in base al contesto. Usa il seguente contesto per rispondere.
    
    Contesto:
    {context}

    Domanda: {question}
    
    Istruzioni:
    1. Fornisci una risposta completa basata sul contesto. Recupera informazioni dal PDF in memoria e raggiungi un livello di dettaglio adeguato.
    2. Se ci sono immagini rilevanti, descrivile
    3. Cita le fonti con [Pagina X] dove X è il numero della pagina
    4. Se non sai rispondere, dì "Non ho informazioni sufficienti"
    5. Se non viene fornito l'argomento preciso nel prompt, riprendi l'argomento del prompt precedente

    Risposta:"""
    
    QA_PROMPT = PromptTemplate.from_template(template)

    # Catena RAG principale
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_groq_llm(),
        retriever=create_retriever(vector_store, selected_files, "text"),
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

    # Aggiunta funzionalità multimodali
    if multimodal:
        def enhanced_query(query: str, include_images: bool = True) -> Dict[str, Any]:
            # Esegui la query testuale normale
            text_result = qa_chain({"query": query})
            
            # Aggiungi risultati immagini se richiesto
            if include_images:
                images = query_images(client, query, selected_files)
                text_result["images"] = images
            
            return text_result
        
        qa_chain.query = enhanced_query

    logger.info("Catena RAG configurata correttamente")
    return qa_chain