# app/pipeline/indexer.py
import os
from langchain_qdrant import Qdrant
from app.utils.pdf_parser import parse_pdf_elements
from app.utils.chunker import chunk_elements
from app.utils.embeddings import get_embedding_model
from app.config import QDRANT_URL, COLLECTION_NAME

def index_document(pdf_path: str):
    """Orchestra la pipeline di indicizzazione personalizzata."""
    print(f"\n--- Inizio indicizzazione per: {os.path.basename(pdf_path)} ---")

    # 1. Parsing granulare
    elements = parse_pdf_elements(pdf_path)

    # 2. Chunking intelligente
    documents = chunk_elements(elements)

    # 3. Inizializzazione modello embedding
    embedding_model = get_embedding_model()

    # 4. Caricamento su Qdrant
    print(f"[*] Caricamento di {len(documents)} chunk su Qdrant (collezione: '{COLLECTION_NAME}')...")
    Qdrant.from_documents(
        documents=documents,
        embedding=embedding_model,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=False, # Imposta a True solo se vuoi cancellare la collezione
    )

    print(f"--- Indicizzazione completata con successo! ---")