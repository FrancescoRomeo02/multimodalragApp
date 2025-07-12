import os
import logging
from typing import List, Dict, Any, Optional

from src.utils.pdf_parser import parse_pdf_elements
from src.utils.qdrant_utils import qdrant_manager
from src.utils.embedder import AdvancedEmbedder
from src.core.models import TextElement, ImageElement, TableElement

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

                all_text_elements.extend(text_elements)
                all_image_elements.extend(image_elements)
                all_table_elements.extend(table_elements)

                processed_files += 1
                logger.info(f"Processed {processed_files}/{len(pdf_paths)}: {os.path.basename(pdf_path)} "
                            f"(texts: {len(text_elements)}, images: {len(image_elements)}, tables: {len(table_elements)})")
            except Exception as e:
                logger.error(f"Error processing file {pdf_path}: {e}")

        if processed_files == 0:
            logger.error("No files processed successfully")
            return False

        success = True

        # Index all elements in batches for efficiency
        indexing_tasks = [
            ("text", all_text_elements, lambda els: self.embedder.embed_documents([el.text for el in els])),
            ("image", all_image_elements, lambda els: [self.embedder.embed_query(el.image_base64) for el in els]),
            ("table", all_table_elements, lambda els: [self.embedder.embed_query(el.table_html) for el in els])
        ]

        for element_type, elements, embedding_func in indexing_tasks:
            if not elements:
                continue
            
            try:
                vectors = embedding_func(elements)
                points = self.qdrant_manager.convert_elements_to_points(elements, vectors)
                
                if self.qdrant_manager.upsert_points(points):
                    logger.info(f"Indexed {len(points)} elements of type {element_type}")
                else:
                    logger.error(f"Failed inserting {element_type} points")
                    success = False
            except Exception as e:
                logger.error(f"Error indexing {element_type}: {e}")
                success = False

        if success:
            logger.info("Indexing completed successfully")
        else:
            logger.warning("Indexing completed with errors")

        return success

    def get_index_status(self) -> Dict[str, Any]:
        try:
            return self.qdrant_manager.health_check()
        except Exception as e:
            logger.error(f"Errore controllo stato indice: {e}")
            return {"error": str(e)}
