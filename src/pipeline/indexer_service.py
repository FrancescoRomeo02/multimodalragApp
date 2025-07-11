import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import base64
import json

from src.utils.pdf_parser import parse_pdf_elements
from src.core.models import TextChunk
from src.utils.qdrant_utils import qdrant_manager
from src.utils.mongodb_utils import mongodb_manager
from src.utils.embedder import AdvancedEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(self, embedder: AdvancedEmbedder, use_semantic_chunking: Optional[bool] = None):
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager
        self.mongodb_manager = mongodb_manager

    def index_files(self, pdf_paths: List[str], force_recreate: bool = False) -> bool:
        if not pdf_paths:
            logger.warning("Nessun file da indicizzare")
            return True

        logger.info(f"Inizio indicizzazione di {len(pdf_paths)} file")

        # Verifica connessione a Qdrant
        if not self.qdrant_manager.verify_connection():
            logger.error("Impossibile connettersi a Qdrant")
            return False
        
        # Verifica connessione a MongoDB
        if not self.mongodb_manager.verify_connection():
            logger.error("Impossibile connettersi a MongoDB")
            return False
        
        # Assicura indici MongoDB
        self.mongodb_manager.ensure_indexes()

        try:
            if force_recreate:
                logger.info("Forzata ricreazione della collezione Qdrant")
                if not self.qdrant_manager.create_collection(
                    embedding_dim=self.embedder.embedding_dim,
                    force_recreate=True
                ):
                    logger.error("Fallita ricreazione collezione Qdrant")
                    return False
            else:
                if not self.qdrant_manager.ensure_collection_exists(self.embedder.embedding_dim):
                    logger.error("Impossibile assicurare esistenza collezione Qdrant")
                    return False
        except Exception as e:
            logger.error(f"Errore gestione collezione Qdrant: {e}")
            return False

  
        processed_files = 0
        all_text_chunks = []

        for pdf_path in pdf_paths:
            try:
                source_filename = os.path.basename(pdf_path)
                
                # Elimina contenuti esistenti prima di riprocessare
                if force_recreate:
                    self.qdrant_manager.delete_by_source(source_filename)
                    self.mongodb_manager.delete_by_source(source_filename)
                
                # Ora parse_pdf_elements restituisce direttamente una lista di TextChunk
                text_chunks_from_parser = parse_pdf_elements(pdf_path)
                
                # Salva i dati originali in MongoDB e aggiorna i TextChunk con i riferimenti
                enhanced_chunks = self._store_original_content_in_mongodb(text_chunks_from_parser, pdf_path)
                
                all_text_chunks.extend(enhanced_chunks)
                
                processed_files += 1
                logger.info(f"Processato {processed_files}/{len(pdf_paths)}: {source_filename} "
                           f"(chunks di testo: {len(enhanced_chunks)})")
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
                    logger.info(f"Indicizzati {len(text_points)} chunk testuali con riferimenti a MongoDB")
                else:
                    logger.error("Fallito inserimento punti in Qdrant")
                    success = False
            except Exception as e:
                logger.error(f"Errore indicizzazione testi: {e}")
                success = False

        if success:
            logger.info("Indicizzazione completata con successo")
        else:
            logger.warning("Indicizzazione completata con errori")

        return success

    def _store_original_content_in_mongodb(self, text_chunks: List[TextChunk], pdf_path: str) -> List[TextChunk]:
        """
        Salva i contenuti originali in MongoDB e aggiorna i TextChunk con i riferimenti.
        
        Args:
            text_chunks: Lista di TextChunk dal parser
            pdf_path: Percorso del file PDF
            
        Returns:
            Lista di TextChunk aggiornati con i riferimenti a MongoDB
        """
        enhanced_chunks = []
        source_filename = os.path.basename(pdf_path)
        
        for chunk in text_chunks:
            # Determina il tipo di contenuto originale dai metadati
            origin_type = chunk.metadata.get("origin_type", "text")
            
            # Prepara i metadati per MongoDB (copia per evitare modifiche indesiderate)
            mongo_metadata = chunk.metadata.copy()
            mongo_metadata["source"] = source_filename
            
            # Memorizza il contenuto in MongoDB in base al tipo
            mongo_id = None
            
            # Logga il tipo di contenuto per debug
            logger.debug(f"Salvataggio in MongoDB: tipo={origin_type}, testo={chunk.text[:50]}...")
            
            if "image" in origin_type.lower():  # Gestisci tutti i tipi di contenuto immagine
                # Per le immagini, potremmo avere i dati dell'immagine nei metadati
                image_base64 = mongo_metadata.get("image_base64", "")
                caption = chunk.text if chunk.text else "Immagine senza descrizione"
                
                if image_base64:
                    mongo_id = self.mongodb_manager.store_image_document(
                        image_base64=image_base64,
                        caption=caption,
                        source=source_filename,
                        page=mongo_metadata.get("page", 0),
                        additional_metadata=mongo_metadata
                    )
                    # Rimuove dati binari pesanti dai metadati di Qdrant
                    if "image_base64" in mongo_metadata:
                        del mongo_metadata["image_base64"]
                    logger.debug(f"Salvata immagine in MongoDB con ID: {mongo_id}")
            
            elif "table" in origin_type.lower():  # Gestisci tutti i tipi di contenuto tabella
                # Per le tabelle, potremmo avere i dati strutturati nei metadati
                table_data = mongo_metadata.get("table_data", {})
                table_markdown = mongo_metadata.get("table_markdown", chunk.text)
                
                mongo_id = self.mongodb_manager.store_table_document(
                    table_markdown=table_markdown,
                    table_data=table_data or {},
                    source=source_filename,
                    page=mongo_metadata.get("page", 0),
                    additional_metadata=mongo_metadata
                )
                # Rimuove dati strutturati pesanti dai metadati di Qdrant
                if "table_data" in mongo_metadata:
                    del mongo_metadata["table_data"]
                logger.debug(f"Salvata tabella in MongoDB con ID: {mongo_id}")
            
            else:  # Per qualsiasi altro tipo, incluso testo normale
                mongo_id = self.mongodb_manager.store_text_document(
                    text=chunk.text,
                    source=source_filename,
                    page=mongo_metadata.get("page", 0),
                    additional_metadata=mongo_metadata
                )
                logger.debug(f"Salvato testo in MongoDB con ID: {mongo_id}")
            
            # Crea una copia del chunk con riferimento a MongoDB
            if mongo_id:
                enhanced_chunk = TextChunk(
                    text=chunk.text,
                    metadata=mongo_metadata,
                    mongo_id=mongo_id
                )
                enhanced_chunks.append(enhanced_chunk)
                logger.debug(f"Chunk aggiornato con mongo_id: {mongo_id}")
            else:
                # Se non è stato possibile salvare in MongoDB, usa il chunk originale ma registra l'errore
                logger.warning(f"Impossibile salvare in MongoDB: tipo={origin_type}, testo={chunk.text[:30]}...")
                enhanced_chunks.append(chunk)
        
        # Statistiche per tipo di contenuto
        content_types = {}
        for chunk in enhanced_chunks:
            origin_type = chunk.metadata.get("origin_type", "unknown")
            content_types[origin_type] = content_types.get(origin_type, 0) + 1
            
        logger.info(f"Salvati {len(enhanced_chunks)} documenti in MongoDB per {source_filename}: {content_types}")
        return enhanced_chunks
    
    def get_index_status(self) -> Dict[str, Any]:
        """
        Restituisce informazioni sullo stato degli indici di entrambi i database.
        """
        try:
            qdrant_status = self.qdrant_manager.health_check()
            mongodb_status = self.mongodb_manager.health_check()
            
            return {
                "qdrant": qdrant_status,
                "mongodb": mongodb_status
            }
        except Exception as e:
            logger.error(f"Errore controllo stato indici: {e}")
            return {"error": str(e)}
