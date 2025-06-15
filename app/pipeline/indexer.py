import os
import qdrant_client
from unstructured.partition.pdf import partition_pdf

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
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
    print(f"\nIndicizzazione PDF: {os.path.basename(pdf_path)}")

    # 1. Setup modello embedding, chunking
    Settings.embed_model = get_embedding_model()
    chunk_size = CHUNK_SIZE or 512
    chunk_overlap = CHUNK_OVERLAP or 50

    # 2. Parsing granulare
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=False,
    )

    # 3. Trasforma in Document con chunking
    docs = []
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for e in elements:
        text = str(e)
        chunks = [text] if len(text) <= chunk_size else splitter.split_text(text)
        for chunk in chunks:
            doc = Document(text=chunk, metadata={
                "page": getattr(e.metadata, "page_number", None),
                "type": type(e).__name__,
                "filename": os.path.basename(pdf_path),
            })
            docs.append(doc)

    print(f"📄 Creati {len(docs)} documenti chunked")

    # 4. Configura Qdrant (hybrid search)
    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        enable_hybrid=True,
        batch_size=20
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Indicizzazione da documents
    index = VectorStoreIndex.from_documents(
        documents=docs,
        storage_context=storage_context,
        show_progress=True
    )
    print("✅ Indicizzazione completata con hybrid search")
    return index

if __name__ == "__main__":
    file = '/Users/fraromeo/Documents/02_Areas/University/LM/LM_24-25/SEM2/MdCS/progetto 2.pdf'
    index_document(file)