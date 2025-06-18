import os
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from app.utils.pdf_parser import parse_pdf_elements
from app.utils.embeddings import get_embedding_model, validate_embedding_dimensions
from app.config import QDRANT_URL, COLLECTION_NAME
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_document_from_element(element: Dict[str, Any], element_type: str) -> Document:
    """
    Crea un documento Langchain da un elemento estratto dal PDF
    """
    if element_type == "text":
        return Document(
            page_content=element["text"],
            metadata=element["metadata"]
        )
    elif element_type == "image":
        return Document(
            page_content=element["page_content"],
            metadata=element["metadata"]
        )
    else:
        raise ValueError(f"Tipo elemento non supportato: {element_type}")

def validate_elements_before_indexing(text_elements: List[Dict], image_elements: List[Dict]) -> bool:
    # Verifica struttura elementi di testo
    for i, elem in enumerate(text_elements):
        if "text" not in elem or "metadata" not in elem:
            logger.error(f"Elemento testo {i} malformato: manca 'text' o 'metadata'")
            return False
        
        metadata = elem["metadata"]
        required_fields = ["type", "source", "page"]
        for field in required_fields:
            if field not in metadata:
                logger.error(f"Elemento testo {i}: manca campo metadata '{field}'")
                return False
    
    # Verifica struttura elementi immagine
    for i, elem in enumerate(image_elements):
        if "image_base64" not in elem or "metadata" not in elem or "page_content" not in elem:
            logger.error(f"Elemento immagine {i} malformato")
            return False
        
        metadata = elem["metadata"]
        required_fields = ["type", "source", "page"]
        for field in required_fields:
            if field not in metadata:
                logger.error(f"Elemento immagine {i}: manca campo metadata '{field}'")
                return False
    
    return True

def index_document(pdf_path: str, force_recreate: bool = False):
    """
    Indicizza un documento PDF con gestione errori
    """
    filename = os.path.basename(pdf_path)
    logger.info(f"\n--- Indicizzazione: {filename} ---")

    try:
        # 1. Verifica esistenza file
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File non trovato: {pdf_path}")

        # 2. Estrazione elementi
        logger.info("Estrazione elementi dal PDF...")
        text_elements, image_elements = parse_pdf_elements(pdf_path)
        logger.info(f"Elementi estratti: {len(text_elements)} testi, {len(image_elements)} immagini")

        # 3. Validazione elementi
        if not validate_elements_before_indexing(text_elements, image_elements):
            raise ValueError("Elementi estratti non validi")

        # 4. Inizializzazione embedder
        logger.info("Inizializzazione modello embedding...")
        embedder = get_embedding_model()

        # 5. Processamento testo
        text_docs = []
        if text_elements:
            logger.info(f"Generazione embedding per {len(text_elements)} testi...")
            
            # Estrai i testi
            texts = [elem["text"] for elem in text_elements]
            
            # Genera embedding
            try:
                embeddings = embedder.embed_documents(texts)
                
                # Valida embedding
                if not validate_embedding_dimensions(embeddings):
                    raise ValueError("Embedding di testo non validi")
                
                # Crea documenti
                for elem in text_elements:
                    text_docs.append(create_document_from_element(elem, "text"))
                
                logger.info(f"Embedding testo generati: {len(embeddings)} di dimensione {len(embeddings[0])}")
                
            except Exception as e:
                logger.error(f"Errore durante l'embedding del testo: {str(e)}")
                raise

        # 6. Processamento immagini
        image_docs = []
        if image_elements:
            logger.info(f"Generazione embedding per {len(image_elements)} immagini...")
            
            try:
                # Genera embedding per le immagini
                embedded_images = embedder.embed_images(image_elements)
                
                # Converti in documenti
                for img in embedded_images:
                    if "embedding" in img and len(img["embedding"]) > 0:
                        image_docs.append(create_document_from_element(img, "image"))
                    else:
                        logger.warning("Immagine senza embedding valido, saltata")
                
                logger.info(f"Immagini processate con successo: {len(image_docs)}/{len(image_elements)}")
                
            except Exception as e:
                logger.error(f"Errore durante l'embedding delle immagini: {str(e)}")
                # Non interrompere il processo se solo le immagini falliscono
                logger.warning("Continuando senza le immagini...")

        # 7. Verifica che ci siano documenti da indicizzare
        all_docs = text_docs + image_docs
        if not all_docs:
            raise ValueError("Nessun documento valido da indicizzare")

        logger.info(f"Documenti totali da indicizzare: {len(all_docs)}")

        # 8. Caricamento su Qdrant
        logger.info("Caricamento documenti in Qdrant...")
        
        try:
            vector_store = Qdrant.from_documents(
                documents=all_docs,
                embedding=embedder,
                url=QDRANT_URL,
                collection_name=COLLECTION_NAME,
                batch_size=16,  # Ridotto per maggiore stabilità
                force_recreate=force_recreate,
                prefer_grpc=True
            )
            
            logger.info("--- Indicizzazione completata con successo! ---")
            return vector_store
            
        except Exception as e:
            logger.error(f"Errore durante il caricamento su Qdrant: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Errore durante l'indicizzazione di {filename}: {str(e)}")
        raise

def index_multiple_documents(pdf_paths: List[str], force_recreate: bool = False):
    logger.info(f"Indicizzazione di {len(pdf_paths)} documenti")
    
    successful = 0
    failed = []
    
    for i, pdf_path in enumerate(pdf_paths):
        try:
            # Per il primo documento, potrebbe essere necessario ricreare la collezione
            force_recreate_current = force_recreate and i == 0
            
            index_document(pdf_path, force_recreate=force_recreate_current)
            successful += 1
            
        except Exception as e:
            logger.error(f"Fallita indicizzazione di {pdf_path}: {str(e)}")
            failed.append(pdf_path)
    
    logger.info(f"Indicizzazione completata: {successful} successi, {len(failed)} fallimenti")
    
    if failed:
        logger.warning(f"Documenti falliti: {failed}")
    
    return successful, failed