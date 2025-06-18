import os
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from app.utils.pdf_parser import parse_pdf_elements
from app.utils.embeddings import get_embedding_model
from app.config import QDRANT_URL, COLLECTION_NAME
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def index_document(pdf_path: str):
    logger.info(f"\n--- Indicizzazione: {os.path.basename(pdf_path)} ---")

    try:
        # 1.Estrazione elementi
        text_elements, image_elements = parse_pdf_elements(pdf_path)
        logger.info(f"Elementi estratti: {len(text_elements)} testi, {len(image_elements)} immagini")

        # 2.Inizializzazione embedder
        embedder = get_embedding_model()

        # 3. Processamento testo
        text_docs = []
        if text_elements:
            logger.info(f"Generazione embedding per {len(text_elements)} testi...")
            texts = [elem["text"] for elem in text_elements]
            embeddings = embedder.embed_documents(texts)
            
            for elem, emb in zip(text_elements, embeddings):
                text_docs.append(Document(
                    page_content=elem["text"],
                    metadata=elem["metadata"],
                    embedding=emb
                ))

        # 4.Processamento immagini
        image_docs = []
        if image_elements:
            logger.info(f"Generazione embedding per {len(image_elements)} immagini...")
            try:
                embedded_images = embedder.embed_images(image_elements)
                successful_images = 0
                
                for img in embedded_images:
                    if "embedding" in img:
                        if len(img["embedding"]) == 1536: 
                            image_docs.append(Document(
                                page_content=img["metadata"].get("description", "Immagine documento"),
                                metadata={
                                    **img["metadata"],
                                    "content_type": "image",
                                    "original_size": len(img["image_base64"])
                                },
                                embedding=img["embedding"]
                            ))
                            successful_images += 1
                        else:
                            logger.warning(f"Embedding immagine con dimensione errata: {len(img['embedding'])}")
                
                logger.info(f"Immagini processate con successo: {successful_images}/{len(image_elements)}")
            except Exception as e:
                logger.error(f"Errore durante l'embedding delle immagini: {str(e)}")
                raise

        # 5. Caricamento su Qdrant
        all_docs = text_docs + image_docs
        logger.info(f"Caricamento {len(all_docs)} documenti in Qdrant...")
        

        Qdrant.from_documents(
            documents=all_docs,
            embedding=embedder, 
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            batch_size=32,
            force_recreate=False,
            prefer_grpc=True
        )
        
        logger.info("--- Indicizzazione completata con successo! ---")

    except Exception as e:
        logger.error(f"Errore durante l'indicizzazione: {str(e)}")
        raise