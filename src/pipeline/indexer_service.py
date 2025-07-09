import os
import logging
from typing import List, Dict, Any, Optional

from src.utils.pdf_parser_unified import parse_pdf_elements, parse_pdf_elements_unified
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
from src.utils.table_info import sanitize_table_cells

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(self, embedder: AdvancedEmbedder, use_semantic_chunking: Optional[bool] = None):
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager
        self.use_semantic_chunking = SEMANTIC_CHUNKING_ENABLED if use_semantic_chunking is None else use_semantic_chunking

        if self.use_semantic_chunking:
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
                tables_dicts = sanitize_table_cells(tables_dicts)

                if self.semantic_chunker:
                    logger.info(f"Applicando chunking semantico per {os.path.basename(pdf_path)}")
                    chunked_texts = self.semantic_chunker.chunk_text_elements(texts_dicts)
                    enriched_texts, enriched_images, enriched_tables = \
                        self.semantic_chunker.associate_media_to_chunks(chunked_texts, images_dicts, tables_dicts)

                    stats = self.semantic_chunker.get_chunking_stats(enriched_texts)
                    if stats['total_chunks'] > 0:
                        logger.info(f"Chunking semantico - Stats: {len(enriched_texts)} chunk totali, "
                                    f"avg: {stats['avg_chunk_length']:.0f} caratteri, "
                                    f"semantici: {stats['semantic_chunks']}, fallback: {stats['fallback_chunks']}")
                    else:
                        logger.info(f"Chunking semantico - Nessun chunk di testo, "
                                    f"ma trovati {len(enriched_images)} immagini e {len(enriched_tables)} tabelle")

                    text_elements = [TextElement(**d) for d in enriched_texts]
                    image_elements = [ImageElement(**d) for d in enriched_images]
                    table_elements = [TableElement(**d) for d in enriched_tables]
                else:
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

    def index_documents_enhanced(self, pdf_paths: List[str], use_vlm: bool = True, groq_api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Indicizza documenti PDF con parser unificato e statistiche avanzate.
        
        Args:
            pdf_paths: Lista dei percorsi ai file PDF
            use_vlm: Se utilizzare l'analisi VLM (default: True)
            groq_api_key: Chiave API Groq (opzionale)
            
        Returns:
            Dict con risultati dell'indicizzazione e statistiche dettagliate
        """
        logger.info(f"Avvio indicizzazione avanzata di {len(pdf_paths)} file PDF")

        all_text_elements: List[TextElement] = []
        all_image_elements: List[ImageElement] = []
        all_table_elements: List[TableElement] = []
        all_stats = []
        processed_files = 0

        for pdf_path in pdf_paths:
            try:
                # Usa il parser unificato per ottenere anche le statistiche
                texts_dicts, images_dicts, tables_dicts, stats = parse_pdf_elements_unified(
                    pdf_path, use_vlm=use_vlm, groq_api_key=groq_api_key
                )
                all_stats.append({
                    'file': os.path.basename(pdf_path),
                    'stats': stats
                })
                
                tables_dicts = sanitize_table_cells(tables_dicts)

                if self.semantic_chunker:
                    logger.info(f"Applicando chunking semantico per {os.path.basename(pdf_path)}")
                    chunked_texts = self.semantic_chunker.chunk_text_elements(texts_dicts)
                    enriched_texts, enriched_images, enriched_tables = \
                        self.semantic_chunker.associate_media_to_chunks(chunked_texts, images_dicts, tables_dicts)

                    chunking_stats = self.semantic_chunker.get_chunking_stats(enriched_texts)
                    if chunking_stats['total_chunks'] > 0:
                        logger.info(f"Chunking semantico - Stats: {len(enriched_texts)} chunk totali, "
                                    f"avg: {chunking_stats['avg_chunk_length']:.0f} caratteri, "
                                    f"semantici: {chunking_stats['semantic_chunks']}, fallback: {chunking_stats['fallback_chunks']}")
                    else:
                        logger.info(f"Chunking semantico - Nessun chunk di testo, "
                                    f"ma trovati {len(enriched_images)} immagini e {len(enriched_tables)} tabelle")

                    all_text_elements.extend([TextElement(text=txt["text"], metadata=txt["metadata"]) 
                                              for txt in enriched_texts])
                    all_image_elements.extend([ImageElement(image_base64=img["image_base64"], 
                                                            page_content=img["page_content"], 
                                                            metadata=img["metadata"]) 
                                               for img in enriched_images])
                    all_table_elements.extend([TableElement(table_data=tbl["table_data"], 
                                                            table_markdown=tbl["table_markdown"], 
                                                            metadata=tbl["metadata"]) 
                                               for tbl in enriched_tables])
                else:
                    all_text_elements.extend([TextElement(text=txt["text"], metadata=txt["metadata"]) 
                                              for txt in texts_dicts])
                    all_image_elements.extend([ImageElement(image_base64=img["image_base64"], 
                                                            page_content=img["page_content"], 
                                                            metadata=img["metadata"]) 
                                               for img in images_dicts])
                    all_table_elements.extend([TableElement(table_data=tbl["table_data"], 
                                                            table_markdown=tbl["table_markdown"], 
                                                            metadata=tbl["metadata"]) 
                                               for tbl in tables_dicts])

                processed_files += 1
                logger.info(f"File {processed_files}/{len(pdf_paths)} processato: {os.path.basename(pdf_path)} "
                          f"(VLM: {'✓' if stats.vlm_analysis_used else '✗'})")

            except Exception as e:
                logger.error(f"Errore nel processamento di {pdf_path}: {str(e)}")
                continue

        if processed_files == 0:
            raise ValueError("Nessun file è stato processato con successo")

        # Salva nel vector store usando la stessa logica dell'index_files esistente
        success = True
        
        if all_text_elements:
            try:
                text_vectors = [self.embedder.embed_query(el.text) for el in all_text_elements]
                text_points = self.qdrant_manager.convert_elements_to_points(all_text_elements, text_vectors)
                if self.qdrant_manager.upsert_points(text_points):
                    logger.info(f"Indicizzati {len(text_points)} elementi di testo")
                else:
                    logger.error("Fallito inserimento punti testo")
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

        # Calcola statistiche aggregate
        total_parsing_time = sum(stat['stats'].total_time for stat in all_stats)
        total_pages = sum(stat['stats'].total_pages for stat in all_stats)
        total_vlm_pages = sum(stat['stats'].vlm_pages_analyzed for stat in all_stats)
        total_fallback_pages = sum(stat['stats'].fallback_pages for stat in all_stats)

        result = {
            'files_processed': processed_files,
            'total_files': len(pdf_paths),
            'indexing_success': success,
            'elements': {
                'texts': len(all_text_elements),
                'images': len(all_image_elements),
                'tables': len(all_table_elements)
            },
            'statistics': {
                'total_parsing_time': total_parsing_time,
                'total_pages': total_pages,
                'vlm_analysis_used': any(stat['stats'].vlm_analysis_used for stat in all_stats),
                'vlm_pages_analyzed': total_vlm_pages,
                'fallback_pages': total_fallback_pages,
                'files_stats': all_stats
            }
        }

        logger.info(f"Indicizzazione completata: {processed_files} file, {len(all_text_elements)} testi, "
                   f"{len(all_image_elements)} immagini, {len(all_table_elements)} tabelle")
        logger.info(f"Analisi VLM: {total_vlm_pages} pagine LLM, {total_fallback_pages} fallback")

        return result
