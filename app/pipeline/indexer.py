import os
import qdrant_client
from unstructured.partition.pdf import partition_pdf

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.utils.embeddings import get_embedding_model
from app.config import (
    QDRANT_URL,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def index_document(pdf_path: str):
    """
    Orchestra un'indicizzazione di massima qualità usando 'unstructured' direttamente
    per il parsing granulare e LlamaIndex per l'indicizzazione, con suddivisione
    dei nodi troppo grandi.
    """
    print(f"\n--- Inizio indicizzazione di MASSIMA QUALITÀ per: {os.path.basename(pdf_path)} ---")

    try:
        # --- 1. Configurazione Globale di LlamaIndex ---
        print("[*] Configurazione dei modelli...")
        Settings.embed_model = get_embedding_model()
        
        # --- 2. Parsing Granulare Diretto con 'unstructured' ---
        print(f"[*] Parsing granulare del PDF con 'unstructured' (strategy: hi_res)...")
        
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False,
            # --- RIMOSSO 'chunking_strategy' per ottenere elementi granulari ---
        )
        print(f"[*] Trovati {len(elements)} elementi strutturali (paragrafi, tabelle, titoli).")

        # --- 3. Trasformazione e Suddivisione Intelligente dei Nodi ---
        print("[*] Trasformazione degli elementi in Nodi e suddivisione dei nodi troppo grandi...")
        
        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        final_nodes = []
        for element in elements:
            if len(str(element)) < CHUNK_SIZE:
                node = TextNode(
                    text=str(element),
                    metadata={
                        "page_number": getattr(element.metadata, 'page_number', None),
                        "element_type": str(type(element).__name__),
                        "filename": str(element.metadata.filename)
                    }
                )
                final_nodes.append(node)
            else:
                temp_doc_text = str(element)
                text_chunks = splitter.split_text(temp_doc_text)
                
                for text_chunk in text_chunks:
                    node = TextNode(
                        text=text_chunk,
                        metadata={
                            "page_number": getattr(element.metadata, 'page_number', None),
                            "element_type": str(type(element).__name__),
                            "filename": str(element.metadata.filename)
                        }
                    )
                    final_nodes.append(node)
        
        print(f"[*] Creati {len(final_nodes)} nodi finali dopo la suddivisione.")

        # --- 4. Connessione a Qdrant ---
        # ... (Questa parte non cambia) ...
        print(f"[*] Connessione a Qdrant...")
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # --- 5. Indicizzazione dei Nodi Finali ---
        print(f"[*] Indicizzazione di {len(final_nodes)} nodi granulari in Qdrant...")
        VectorStoreIndex(
            nodes=final_nodes,
            storage_context=storage_context,
            show_progress=True
        )

        print(f"--- Indicizzazione di massima qualità completata con successo! ---")

    except Exception as e:
        print(f"[!!!] Errore critico durante l'indicizzazione: {e}")
        raise e