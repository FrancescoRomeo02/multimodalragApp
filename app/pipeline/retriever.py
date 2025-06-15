# app/pipeline/retriever.py

import qdrant_client
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from app.utils.embeddings import get_embedding_model
from app.config import QDRANT_URL, COLLECTION_NAME

def create_query_engine():
    """
    Crea e restituisce un motore di query (query engine) connesso
    all'indice Qdrant esistente, configurato per la ricerca ibrida.
    """
    print("[*] Configurazione del motore di query...")

    # 1. Configura il modello di embedding (deve essere lo stesso usato per l'indicizzazione)
    Settings.embed_model = get_embedding_model()

    # 2. Connettiti al Vector Store Qdrant esistente
    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)

    # 3. Carica l'indice dal Vector Store
    # Non stiamo creando un nuovo indice, ma caricando quello esistente.
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # 4. Crea il Query Engine
    # Questo è il cuore del retrieval.
    query_engine = index.as_query_engine(
        # Specifichiamo la modalità di ricerca ibrida!
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        # Quanti risultati recuperare. 2-3 è un buon punto di partenza.
        similarity_top_k=3, 
    )

    print("[*] Motore di query pronto.")
    return query_engine