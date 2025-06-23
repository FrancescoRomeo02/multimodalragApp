from typing import List, Optional, Dict, Any
from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import qdrant_client
from langchain.schema.retriever import BaseRetriever
from qdrant_client.http import models
from pydantic import BaseModel
from app.llm.groq_client import get_groq_llm
from app.config import QDRANT_URL, COLLECTION_NAME
import logging
from app.utils.embedder import get_multimodal_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageResult(BaseModel):
    image_base64: str
    metadata: dict
    score: float
    page_content: str

class RetrievalResult(BaseModel):
    answer: str
    source_documents: List[Dict]
    images: Optional[List[ImageResult]] = None
    confidence_score: Optional[float] = None

def create_content_filter(query_type: Optional[str] = None) -> Optional[models.Filter]:
    """
    Crea filtri per tipo di contenuto usando i campi STANDARDIZZATI
    """
    if query_type == "image":
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.type",
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

def create_retriever(vector_store: Qdrant, 
                    selected_files: List[str] = None, 
                    query_type: Optional[str] = None,
                    k: int = 5) -> BaseRetriever:
    """
    Crea retriever con filtri appropriati
    """
    qdrant_filter = build_combined_filter(selected_files, query_type)
    
    search_kwargs = {
        "k": k,
        "fetch_k": k * 4,  # Fetch più documenti per MMR
        "score_threshold": 0.3,  # Soglia di rilevanza
    }
    
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter
    
    return vector_store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance per diversità
        search_kwargs=search_kwargs
    )

def query_images(client: qdrant_client.QdrantClient, 
                query: str, 
                selected_files: List[str] = None, 
                top_k: int = 3) -> List[ImageResult]:
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
            score_threshold=0.3
        )
        
        image_results = []
        for result in results:
            try:
                metadata = result.payload.get("metadata", {})
                image_base64 = metadata.get("image_base64", "")
                
                if image_base64:
                    image_results.append(ImageResult(
                        image_base64=image_base64,
                        metadata=metadata,
                        score=result.score,
                        page_content=result.payload.get("page_content", "")
                    ))
            except Exception as e:
                logger.warning(f"Errore processamento risultato immagine: {str(e)}")
                continue
        
        return image_results
        
    except Exception as e:
        logger.error(f"Errore durante la query di immagini: {str(e)}")
        return []

def create_enhanced_prompt_template() -> PromptTemplate:
    """Template migliorato per risposte più esaustive e strutturate"""
    template = """Sei un esperto analista che fornisce risposte dettagliate basate sui documenti forniti. 
Segui scrupolosamente queste linee guida:

CONTESTO FORNITO:
{context}

DOMANDA UTENTE:
{question}

DIRETTIVE PER LA RISPOSTA:
1. ANALISI DEL CONTESTO:
   - Valuta attentamente se il contesto contiene informazioni sufficienti e rilevanti
   - Se il contesto è insufficiente, rispondi: "Non ho informazioni sufficienti per rispondere in modo completo."

2. STRUTTURA DELLA RISPOSTA:
   - Introduzione: inquadra brevemente l'argomento
   - Corpo principale: elenca tutti i punti rilevanti in ordine logico
   - Conclusioni: sintesi e eventuali limitazioni
   - Riferimenti: [Fonte: nome_file, Pagina X] per ogni informazione

3. STILE:
   - Usa paragrafi ben strutturati
   - Elenca i punti con numerazione o bullet points quando appropriato
   - Mantieni un tono professionale ma chiaro

4. CONTENUTO:
   - Sii il più esaustivo possibile senza ripetizioni
   - Integra informazioni correlate quando utile
   - Se ci sono immagini/tabelle rilevanti, descrivile brevemente

5. ONESTÀ INTELLETTUALE:
   - Non inventare informazioni
   - Se qualcosa non è chiaro nel contesto, ammettilo
   - Differenzia tra fatti certi e inferenze logiche

RISPOSTA DETTAGLIATA:"""
    
    return PromptTemplate.from_template(template)

def create_rag_chain(selected_files: List[str] = None, multimodal: bool = False):
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
    prompt = create_enhanced_prompt_template()
    
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
                      selected_files: List[str] = None, 
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
               selected_files: List[str] = None,
               multimodal: bool = False) -> List[RetrievalResult]:
    logger.info(f"Esecuzione batch di {len(queries)} query")
    
    results = []
    rag_chain = create_rag_chain(selected_files, multimodal)
    
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

def debug_query_results(query: str, selected_files: List[str] = None, top_k: int = 10):
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
            print(f"Tipo: {result.payload.get('metadata', {}).get('type', 'N/A')}")
            print(f"Fonte: {result.payload.get('metadata', {}).get('source', 'N/A')}")
            print(f"Pagina: {result.payload.get('metadata', {}).get('page', 'N/A')}")
            print(f"Contenuto: {result.payload.get('page_content', '')[:200]}...")
            
    except Exception as e:
        print(f"Errore debug: {str(e)}")

def search_similar_documents(query: str, 
                           selected_files: List[str] = None,
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
            doc_info = {
                "content": result.payload.get("page_content", ""),
                "metadata": result.payload.get("metadata", {}),
                "score": result.score,
                "source": result.payload.get("metadata", {}).get("source", "Unknown"),
                "page": result.payload.get("metadata", {}).get("page", "N/A")
            }
            similar_docs.append(doc_info)
        
        return similar_docs
        
    except Exception as e:
        logger.error(f"Errore nella ricerca di documenti simili: {str(e)}")
        return []

#FUNZIONE PER RECUPERARE DOCUMENTI PER FILE SPECIFICO --> FUNZIONA
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
        
        scroll_filter = models.Filter(must=filters)
        
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
            doc_info = {
                "content": result.payload.get("page_content", ""),
                "metadata": result.payload.get("metadata", {}),
                "id": result.id,
                "source": result.payload.get("metadata", {}).get("source", "Unknown"),
                "page": result.payload.get("metadata", {}).get("page", "N/A"),
                "type": result.payload.get("metadata", {}).get("type", "unknown")
            }
            documents.append(doc_info)
        
        return documents
        
    except Exception as e:
        logger.error(f"Errore nel recupero documenti per fonte {source_file}: {str(e)}")
        return []

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
