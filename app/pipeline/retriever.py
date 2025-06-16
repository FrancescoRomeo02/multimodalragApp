# app/pipeline/retriever.py
from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import qdrant_client
from langchain.schema.retriever import BaseRetriever
from qdrant_client.http import models

from app.utils.embeddings import get_embedding_model
from app.llm.groq_client import get_groq_llm
from app.config import QDRANT_URL, COLLECTION_NAME

def create_filtered_retriever(vector_store: Qdrant, selected_files: list[str]) -> BaseRetriever:
    """Crea un retriever che filtra i risultati in base ai file selezionati."""
    
    # Se non ci sono file selezionati, non possiamo creare un filtro valido.
    # Restituiamo un retriever che non troverà nulla.
    if not selected_files:
        return vector_store.as_retriever(search_kwargs={"k": 0})

    # --- Creazione del Filtro per Qdrant ---
    # Questo è il cuore della modifica. Creiamo una condizione di filtro.
    # Vogliamo i documenti dove il campo 'filename' nei metadati
    # corrisponde a UNO QUALSIASI dei file selezionati.
    qdrant_filter = models.Filter(
        should=[
            models.FieldCondition(
                key="metadata.filename", # Nota: LangChain aggiunge 'metadata.' al nome del campo
                match=models.MatchValue(value=filename),
            )
            for filename in selected_files
        ]
    )

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "filter": qdrant_filter # <-- APPLICHIAMO IL FILTRO!
        }
    )

def create_rag_chain(selected_files: list[str]):
    """Crea la catena RAG completa per interrogare i documenti."""
    print(f"[*] Creazione catena RAG per i file: {selected_files}")

    # Connettiti a Qdrant
    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    embedding_model = get_embedding_model()
    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
    )

    # Crea il retriever
    retriever = create_filtered_retriever(vector_store, selected_files)


    # Definisci il prompt template
    template = (
        "Sei un assistente di ricerca accademico, preciso e fattuale. La tua missione è rispondere alla domanda basandoti ESCLUSIVAMENTE sul contesto fornito.\n"
        "Analizza attentamente le seguenti informazioni di contesto estratte da un documento scientifico.\n\n"
        "---------------------\n"
        "Contesto:\n{context}\n"
        "---------------------\n\n"
        "Istruzioni per la risposta:\n"
        "1. Formula una risposta concisa e diretta alla domanda: '{question}'\n"
        "2. Non usare alcuna conoscenza esterna al contesto fornito. Se le informazioni non sono presenti, rispondi ESATTAMENTE con la frase: 'Le informazioni fornite non sono sufficienti per rispondere a questa domanda.'\n"
        "3. Per OGNI affermazione che fai, cita la fonte indicando il numero di pagina tra parentesi. Esempio: 'Il composto ha mostrato un'efficacia del 95% (Pagina 7).'\n"
        "4. Se più fonti supportano un'affermazione, puoi raggrupparle. Esempio: 'Diversi esperimenti confermano questo risultato (Pagina 4, Pagina 9).'\n\n"
        "Risposta Accademica:"
    )
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