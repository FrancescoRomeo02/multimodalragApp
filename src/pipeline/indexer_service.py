import os
import logging
from typing import List, Dict, Any, Union

from src.utils.pdf_parser import parse_pdf_elements
from src.utils.qdrant_utils import qdrant_manager
from src.utils.embedder import AdvancedEmbedder
from src.core.models import TextElement, ImageElement, TableElement

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Gestisce l'indicizzazione di documenti PDF in Qdrant.
    Utilizza QdrantManager per tutte le operazioni sul database vettoriale.
    Supporta testo, immagini e tabelle.
    """

    def __init__(self, embedder: AdvancedEmbedder):
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager

    def index_files(self, pdf_paths: List[str], force_recreate: bool = False) -> bool:
        if not pdf_paths:
            logger.warning("Nessun file da indicizzare")
            return True

        logger.info(f"Inizio indicizzazione di {len(pdf_paths)} file")

        # Verifica connessione Qdrant
        if not self.qdrant_manager.verify_connection():
            logger.error("Impossibile connettersi a Qdrant")
            return False

        # Gestione creazione/ricreazione collezione
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

        # Liste di elementi per tipo
        all_text_elements: List[TextElement] = []
        all_image_elements: List[ImageElement] = []
        all_table_elements: List[TableElement] = []

        processed_files = 0

        for pdf_path in pdf_paths:
            try:
                texts_dicts, images_dicts, tables_dicts = parse_pdf_elements(pdf_path)

                # Crea istanze modelli Pydantic per ciascun tipo
                text_elements = [TextElement(**d) for d in texts_dicts]
                image_elements = [ImageElement(**d) for d in images_dicts]
                table_elements = [TableElement(**d) for d in tables_dicts]

                all_text_elements.extend(text_elements)
                all_image_elements.extend(image_elements)
                all_table_elements.extend(table_elements)

                processed_files += 1
                logger.info(f"Processato {processed_files}/{len(pdf_paths)}: {os.path.basename(pdf_path)} "
                            f"(testi: {len(text_elements)}, immagini: {len(image_elements)}, tabelle: {len(table_elements)})")

            except Exception as e:
                logger.error(f"Errore processamento file {pdf_path}: {e}")

        if processed_files == 0:
            logger.error("Nessun file processato con successo")
            return False

        success = True

        # Indicizzazione testi
        if all_text_elements:
            try:
                text_vectors = [self.embedder.embed_query(el.text) for el in all_text_elements]
                text_points = self.qdrant_manager.convert_elements_to_points(all_text_elements, text_vectors)
                if self.qdrant_manager.upsert_points(text_points):
                    logger.info(f"Indicizzati {len(text_points)} elementi testuali")
                else:
                    logger.error("Fallito inserimento punti testuali")
                    success = False
            except Exception as e:
                logger.error(f"Errore indicizzazione testi: {e}")
                success = False

        # Indicizzazione immagini e descrizioni)
        if all_image_elements:
            try:
                image_vectors = [self.embedder.embed_query(el.image_base64) for el in all_image_elements]
                image_points = self.qdrant_manager.convert_elements_to_points(all_image_elements, image_vectors)
                if self.qdrant_manager.upsert_points(image_points):
                    logger.info(f"Indicizzate {len(image_points)} immagini")
                else:
                    logger.error("Fallito inserimento punti immagini")
                    success = False

                # Indicizzazione descrizioni immagini
                images_descriptions_vectors = [self.embedder.embed_query(el.metadata.image_caption or el.metadata.manual_caption or "")
                                               for el in all_image_elements]
                images_descriptions_points = self.qdrant_manager.convert_elements_to_points(
                    all_image_elements, images_descriptions_vectors)
                if self.qdrant_manager.upsert_points(images_descriptions_points):
                    logger.info(f"Indicizzate {len(images_descriptions_points)} descrizioni immagini")
                else:
                    logger.error("Fallito inserimento punti descrizioni immagini")
                    success = False

            except Exception as e:
                logger.error(f"Errore indicizzazione immagini e descrizioni: {e}")
                success = False

        # Indicizzazione tabelle
        if all_table_elements:
            try:
                table_texts = [el.table_markdown for el in all_table_elements]
                table_vectors = [self.embedder.embed_query(txt) for txt in table_texts]
                table_points = self.qdrant_manager.convert_elements_to_points(all_table_elements, table_vectors)
                if self.qdrant_manager.upsert_points(table_points):
                    logger.info(f"Indicizzate {len(table_points)} tabelle")
                else:
                    logger.error("Fallito inserimento punti tabelle")
                    success = False
            except Exception as e:
                logger.error(f"Errore indicizzazione tabelle: {e}")
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
