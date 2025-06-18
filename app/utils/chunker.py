# app/utils/chunker.py
from typing import List
from langchain_core.documents import Document
from unstructured.documents.elements import (
    Element, Title, NarrativeText, Image, 
    Table, Header, Footer, PageBreak,
)
from app.config import CHUNK_SIZE

def chunk_elements(elements: List[Element]) -> List[Document]:
    """""
    1. Elimina elementi inutili (footer, page breaks)
    2. Unisce elementi correlati (titoli + contenuto)
    3. Crea chunk semanticamente coerenti
    """
    print("[*] Avanzato chunking degli elementi...")
    
    # Filtra elementi inutili
    filtered_elements = [
        el for el in elements 
        if not isinstance(el, (Footer, PageBreak))
    ]
    
    chunks = []
    current_chunk = []
    current_metadata = {}
    min_chunk_size = CHUNK_SIZE // 3  # Soglia minima per evitare chunk troppo piccoli
    
    for element in filtered_elements:
        # Estrai testo e metadati
        text = str(element).strip()
        if not text:
            continue
            
        metadata = {
            "page_number": getattr(element.metadata, 'page_number', None),
            "element_type": type(element).__name__,
            "filename": getattr(element.metadata, 'filename', None),
        }
        
        # Gestione speciale immagini
        if isinstance(element, Image):
            chunks.append(Document(
                page_content="[IMAGE] " + metadata.get("filename", ""),
                metadata={**metadata, "content_type": "image"}
            ))
            continue
            
        # Unisci titoli con il contenuto successivo
        if isinstance(element, (Title, Header)):
            if current_chunk and len(' '.join(current_chunk)) >= min_chunk_size:
                chunks.append(create_chunk(current_chunk, current_metadata))
                current_chunk = []
            current_metadata = metadata
            current_chunk.append(f"# {text}")
        # Unisci testo
        elif isinstance(element, NarrativeText):
            if not current_chunk:
                current_metadata = metadata
            current_chunk.append(text)
        # Gestione tabelle
        elif isinstance(element, Table):
            if current_chunk:
                chunks.append(create_chunk(current_chunk, current_metadata))
                current_chunk = []
            chunks.append(Document(
                page_content=f"[TABLE]\n{text}",
                metadata={**metadata, "content_type": "table"}
            ))
    
    # Aggiungi l'ultimo chunk se supera la soglia minima
    if current_chunk and len(' '.join(current_chunk)) >= min_chunk_size:
        chunks.append(create_chunk(current_chunk, current_metadata))
    
    print(f"[*] Creati {len(chunks)} chunk significativi (da {len(elements)} elementi originali)")
    return chunks

def create_chunk(text_parts: List[str], metadata: dict) -> Document:
    content = '\n\n'.join(text_parts)
    return Document(page_content=content, metadata=metadata)