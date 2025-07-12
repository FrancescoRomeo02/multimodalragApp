import os
import logging
from typing import List, Dict, Any, Optional

from src.utils.pdf_parser import parse_pdf_elements
from src.utils.qdrant_utils import qdrant_manager
from src.utils.embedder import AdvancedEmbedder
from src.core.models import TextElement, ImageElement, TableElement

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(self, embedder: AdvancedEmbedder):
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager
        self.semantic_chunker = None

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

        all_text_elements: List[TextElement] = []
        all_image_elements: List[ImageElement] = []
        all_table_elements: List[TableElement] = []
        processed_files = 0

        for pdf_path in pdf_paths:
            try:
                texts_dicts, images_dicts, tables_dicts = parse_pdf_elements(pdf_path)
  
                text_elements = [TextElement(text=d['text'].text, metadata=d['metadata']) for d in texts_dicts]
                image_elements = [ImageElement(image_base64=d['image_base64'], metadata=d['metadata']) for d in images_dicts]
                table_elements = [TableElement(table_html=d['table_html'], metadata=d['metadata']) for d in tables_dicts]

                for el in table_elements:
                    print(f"Tabella trovata: {el.metadata.table_id} in {el.metadata.source}, pagina {el.metadata.page}")
                    print(f"Contenuto tabella: {el.table_html[:100]}...")  # Stampa i primi 100 caratteri
                    print("-" * 50)

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

        if all_text_elements:
            try:
                text_vectors = self.embedder.embed_documents([el.text for el in all_text_elements])
                text_points = self.qdrant_manager.convert_elements_to_points(all_text_elements, text_vectors)
                if self.qdrant_manager.upsert_points(text_points):
                    logger.info(f"Indicizzati {len(text_points)} elementi testuali")
                else:
                    logger.error("Fallito inserimento punti testuali")
                    success = False
            except Exception as e:
                logger.error(f"Errore indicizzazione testi: {e}")
                success = False

        if all_image_elements:
            try:
                image_vectors = [self.embedder.embed_query(el.image_base64) for el in all_image_elements]
                image_points = self.qdrant_manager.convert_elements_to_points(all_image_elements, image_vectors)
                if self.qdrant_manager.upsert_points(image_points):
                    logger.info(f"Indicizzate {len(image_points)} immagini")
                else:
                    logger.error("Fallito inserimento punti immagini")
                    success = False
            except Exception as e:
                logger.error(f"Errore indicizzazione immagini: {e}")
                success = False

        if all_table_elements:
            try:
                table_texts = [el.table_html for el in all_table_elements]
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
