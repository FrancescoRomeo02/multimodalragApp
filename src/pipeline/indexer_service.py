import os
import logging
from typing import List, Dict, Any, Optional

from src.utils.pdf_parser import parse_pdf_elements, TextChunk
from src.utils.qdrant_utils import qdrant_manager
from src.utils.embedder import AdvancedEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(self, embedder: AdvancedEmbedder, use_semantic_chunking: Optional[bool] = None):
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager

    def index_files(self, pdf_paths: List[str], force_recreate: bool = False) -> bool:
        if not pdf_paths:
            logger.warning("Nessun file da indicizzare")
            return True

        logger.info(f"Inizio indicizzazione di {len(pdf_paths)} file")

        if not self.qdrant_manager.verify_connection():
            logger.error("Impossibile connettersi a Qdrant")
            return False

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

  
        processed_files = 0
        all_text_chunks = []

        for pdf_path in pdf_paths:
            try:
                # Ora parse_pdf_elements restituisce direttamente una lista di TextChunk
                text_chunks = parse_pdf_elements(pdf_path)
                all_text_chunks.extend(text_chunks)
                
                processed_files += 1
                logger.info(f"Processato {processed_files}/{len(pdf_paths)}: {os.path.basename(pdf_path)} "
                           f"(chunks di testo: {len(text_chunks)})")
            except Exception as e:
                logger.error(f"Errore processamento file {pdf_path}: {e}")

        if processed_files == 0:
            logger.error("Nessun file processato con successo")
            return False

        success = True

        # Ora gestiamo tutti i chunk di testo in modo unificato
        if all_text_chunks:
            try:
                # Creiamo embedding per tutti i TextChunk
                texts_for_embedding = [chunk.text for chunk in all_text_chunks]
                text_vectors = self.embedder.embed_documents(texts_for_embedding)
                
                # Convertire i TextChunk in punti Qdrant
                text_points = self.qdrant_manager.convert_text_chunks_to_points(all_text_chunks, text_vectors)
                
                if self.qdrant_manager.upsert_points(text_points):
                    logger.info(f"Indicizzati {len(text_points)} chunk testuali")
                else:
                    logger.error("Fallito inserimento punti")
                    success = False
            except Exception as e:
                logger.error(f"Errore indicizzazione testi: {e}")
                success = False

        if success:
            logger.info("Indicizzazione completata con successo")
        else:
            logger.warning("Indicizzazione completata con errori")

        return success

    def get_index_status(self) -> Dict[str, Any]:
        try:
            return self.qdrant_manager.health_check()
        except Exception as e:
            logger.error(f"Errore controllo stato indice: {e}")
            return {"error": str(e)}
