import os
import logging
from typing import List, Dict, Any, Union, Optional

from src.utils.pdf_parser import parse_pdf_elements
from src.utils.qdrant_utils import qdrant_manager
from src.utils.embedder import AdvancedEmbedder
from src.utils.semantic_chunker import MultimodalSemanticChunker
from src.core.models import TextElement, ImageElement, TableElement
from src.config import (
    SEMANTIC_CHUNK_SIZE, 
    SEMANTIC_CHUNK_OVERLAP, 
    SEMANTIC_THRESHOLD, 
    MIN_CHUNK_SIZE,
    CHUNKING_EMBEDDING_MODEL,
    SEMANTIC_CHUNKING_ENABLED
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Gestisce l'indicizzazione di documenti PDF in Qdrant.
    Utilizza QdrantManager per tutte le operazioni sul database vettoriale.
    Supporta testo, immagini e tabelle con chunking semantico avanzato.
    """

    def __init__(self, embedder: AdvancedEmbedder, use_semantic_chunking: Optional[bool] = None):
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager
        
        # Usa configurazione globale se non specificato
        if use_semantic_chunking is None:
            use_semantic_chunking = SEMANTIC_CHUNKING_ENABLED
            
        self.use_semantic_chunking = use_semantic_chunking
        
        # Inizializza il chunker semantico se abilitato
        if use_semantic_chunking:
            try:
                self.semantic_chunker = MultimodalSemanticChunker(
                    embedding_model=CHUNKING_EMBEDDING_MODEL,
                    chunk_size=SEMANTIC_CHUNK_SIZE,
                    chunk_overlap=SEMANTIC_CHUNK_OVERLAP,
                    semantic_threshold=SEMANTIC_THRESHOLD,
                    min_chunk_size=MIN_CHUNK_SIZE
                )
                logger.info("Chunker semantico inizializzato con successo")
            except Exception as e:
                logger.warning(f"Errore inizializzazione chunker semantico: {e}")
                logger.warning("Fallback a chunking classico")
                self.semantic_chunker = None
                self.use_semantic_chunking = False
        else:
            self.semantic_chunker = None
            logger.info("Chunking semantico disabilitato, uso metodo classico")

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

                # Usa sempre chunking semantico se disponibile, altrimenti classico
                if self.semantic_chunker:
                    logger.info(f"Applicando chunking semantico per {os.path.basename(pdf_path)}")
                    
                    # Chunking semantico del testo
                    chunked_texts = self.semantic_chunker.chunk_text_elements(texts_dicts)
                    
                    # Associa elementi multimediali ai chunk
                    enriched_texts, enriched_images, enriched_tables = \
                        self.semantic_chunker.associate_media_to_chunks(
                            chunked_texts, images_dicts, tables_dicts
                        )
                    
                    # Statistiche chunking
                    stats = self.semantic_chunker.get_chunking_stats(enriched_texts)
                    logger.info(f"Chunking semantico - Stats: {len(enriched_texts)} chunk totali, "
                               f"avg: {stats['avg_chunk_length']:.0f} caratteri, "
                               f"semantici: {stats['semantic_chunks']}, fallback: {stats['fallback_chunks']}")
                    
                    # Crea istanze modelli Pydantic
                    text_elements = [TextElement(**d) for d in enriched_texts]
                    image_elements = [ImageElement(**d) for d in enriched_images] 
                    table_elements = [TableElement(**d) for d in enriched_tables]
                else:
                    # Fallback al comportamento precedente (senza chunking)
                    logger.info(f"Chunking semantico non disponibile per {os.path.basename(pdf_path)}, uso metodo classico")
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

        # Indicizzazione immagini
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
