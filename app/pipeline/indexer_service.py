import os
import logging
import uuid
from typing import List, Dict, Any

from app.utils.pdf_utils import prepare_elements_from_file, prepare_image_data, create_text_points, create_image_points
from app.utils.qdrant_utils import qdrant_manager
from app.utils.embedder import AdvancedEmbedder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Gestisce l'indicizzazione di documenti PDF in Qdrant.
    Ora utilizza QdrantManager per tutte le operazioni sul database vettoriale.
    """
    
    def __init__(self, embedder: AdvancedEmbedder):
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager

    def index_files(self, pdf_paths: List[str], force_recreate: bool = False) -> bool:
        """
        Indicizza file PDF con opzione di ricreare la collezione
        
        Args:
            pdf_paths: Lista percorsi file PDF da indicizzare
            force_recreate: Se True, ricrea la collezione eliminando quella esistente
            
        Returns:
            bool: True se l'indicizzazione è completata con successo
        """
        if not pdf_paths:
            logger.warning("Nessun file da indicizzare")
            return True

        logger.info(f"Inizio indicizzazione di {len(pdf_paths)} file")

        # Verifica connessione Qdrant
        if not self.qdrant_manager.verify_connection():
            logger.error("Impossibile connettersi a Qdrant")
            return False

        # Gestisci creazione/ricreazione collezione
        try:
            if force_recreate:
                logger.info("Forzata ricreazione della collezione")
                if not self.qdrant_manager.create_collection(
                    embedding_dim=self.embedder.embedding_dim,
                    force_recreate=True
                ):
                    logger.error("Fallita ricreazione collezione")
                    return False
            else:
                if not self.qdrant_manager.ensure_collection_exists(self.embedder.embedding_dim):
                    logger.error("Impossibile assicurare esistenza collezione")
                    return False
        except Exception as e:
            logger.error(f"Errore gestione collezione: {e}")
            return False

        # Processamento e raccolta dati da tutti i file
        all_texts = []
        all_text_metadatas = []
        all_image_descriptions = []
        processed_files = 0

        for path in pdf_paths:
            try:
                texts, images = prepare_elements_from_file(path)
                
                # Processa elementi testuali
                if texts:
                    all_texts.extend(elem.text for elem in texts)
                    all_text_metadatas.extend(elem.metadata.model_dump() for elem in texts)
                
                # Processa elementi immagine
                for elem in images:
                    try:
                        image_data = prepare_image_data(elem)
                        all_image_descriptions.append(image_data)
                    except Exception as e:
                        logger.error(f"Errore processamento immagine da {path}: {e}")
                        continue
                
                processed_files += 1
                logger.info(f"Processato file {processed_files}/{len(pdf_paths)}: {os.path.basename(path)}")
                        
            except Exception as e:
                logger.error(f"Errore processamento file {path}: {e}")
                continue

        if processed_files == 0:
            logger.error("Nessun file processato con successo")
            return False

        logger.info(f"Raccolti {len(all_texts)} testi e {len(all_image_descriptions)} immagini")

        # Indicizzazione elementi testuali
        success_text = True
        if all_texts:
            try:
                text_points, success_text = create_text_points(all_texts, all_text_metadatas, embedder=self.embedder)
                
                if success_text and text_points:
                    if self.qdrant_manager.upsert_points(text_points):
                        logger.info(f"Indicizzati {len(text_points)} elementi di testo")
                    else:
                        logger.error("Fallito inserimento punti testo")
                        success_text = False
                else:
                    logger.error("Fallita creazione punti testo")
                    success_text = False
                    
            except Exception as e:
                logger.error(f"Errore indicizzazione testo: {e}")
                success_text = False

        # Indicizzazione elementi immagine
        success_images = True
        if all_image_descriptions:
            try:
                image_points, success_images = create_image_points(all_image_descriptions, embedder=self.embedder)

                image_text_points, success_text_images = create_text_points(
                    [desc["text"] for desc in all_image_descriptions],
                    [desc["metadata"] for desc in all_image_descriptions],
                    embedder=self.embedder
                )
                
                if success_images and image_points:
                    if self.qdrant_manager.upsert_points(image_points):
                        logger.info(f"Indicizzate {len(image_points)} immagini")
                    else:
                        logger.error("Fallito inserimento punti immagine")
                        success_images = False
                else:
                    logger.error("Fallita creazione punti immagine")
                    success_images = False

                if success_text_images and image_text_points:
                    if self.qdrant_manager.upsert_points(image_text_points):
                        logger.info(f"Indicizzate {len(image_text_points)} descrizioni immagini")
                    else:
                        logger.error("Fallito inserimento punti descrizioni immagini")
                        success_images = False
                else:
                    logger.error("Fallita creazione punti descrizioni immagini")
                    success_images = False
                    
            except Exception as e:
                logger.error(f"Errore indicizzazione immagini: {e}")
                success_images = False

        # Risultato finale
        overall_success = success_text and success_images
        if overall_success:
            logger.info("Processo di indicizzazione completato con successo")
        else:
            logger.warning("Processo di indicizzazione completato con alcuni errori")

        return overall_success

    def get_index_status(self) -> Dict[str, Any]:
        """
        Restituisce informazioni sullo stato dell'indice
        
        Returns:
            Dict con informazioni sulla collezione e connessione
        """
        try:
            return self.qdrant_manager.health_check()
        except Exception as e:
            logger.error(f"Errore controllo stato indice: {e}")
            return {"error": str(e)}