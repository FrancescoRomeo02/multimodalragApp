# app/utils/chunker.py
from typing import List
from unstructured.documents.elements import Element
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_elements(elements: List[Element]) -> List[Document]:
    """
    Trasforma elementi di 'unstructured' in documenti di LangChain,
    e divide quelli troppo grandi.
    """
    print("[*] Trasformazione elementi in documenti e chunking...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    final_docs = []
    for element in elements:
        text = str(element)
        metadata = {
            "page_number": getattr(element.metadata, 'page_number', None),
            "element_type": type(element).__name__,
            "filename": getattr(element.metadata, 'filename', None)
        }
        
        if len(text) <= CHUNK_SIZE:
            doc = Document(page_content=text, metadata=metadata)
            final_docs.append(doc)
        else:
            # Se l'elemento è troppo grande, lo splittiamo
            chunks = splitter.split_text(text)
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata=metadata.copy()) # Usa .copy()
                final_docs.append(doc)

    print(f"[*] Creati {len(final_docs)} documenti finali (chunk).")
    return final_docs