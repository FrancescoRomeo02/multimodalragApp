# app/pipeline/retriever.py
from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import qdrant_client

from app.utils.embeddings import get_embedding_model
from app.llm.groq_client import get_groq_llm
from app.config import QDRANT_URL, COLLECTION_NAME, EMBEDDING_DIM

def create_rag_chain():
    """Crea la catena RAG completa per interrogare i documenti."""
    print("[*] Creazione della catena RAG...")

    # Connettiti a Qdrant
    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    embedding_model = get_embedding_model()
    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
    )

    # Crea il retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "score_threshold": 0.7}
    )

    # Definisci il prompt template
    template = """Usa le seguenti informazioni di contesto per rispondere alla domanda.
    Se non conosci la risposta, dì semplicemente che non la conosci, non cercare di inventare una risposta.
    Rispondi in modo conciso e cita sempre il numero di pagina se disponibile nei metadati.

    Contesto: {context}

    Domanda: {question}

    Risposta utile:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Inizializza l'LLM
    llm = get_groq_llm()

    # Crea la catena RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    print("[*] Catena RAG pronta.")
    return qa_chain