#!/usr/bin/env python3
"""
Script per re-indicizzare i documenti con il nuovo sistema di estrazione contesto
"""

import sys
import logging
from pathlib import Path

# Aggiungi il percorso del modulo principale
sys.path.append(str(Path(__file__).parent))

from app.pipeline.indexer_service import DocumentIndexer
from app.utils.embedder import get_multimodal_embedding_model
from app.utils.qdrant_utils import qdrant_manager
from app.config import COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reindex_with_context():
    """
    Re-indicizza tutti i documenti nella cartella data/raw con il nuovo sistema
    """
    logger.info("Inizio re-indicizzazione con estrazione contesto...")
    
    try:
        # Inizializza i servizi
        embedder = get_multimodal_embedding_model()
        indexer = DocumentIndexer(embedder=embedder)
        
        # Elimina la collezione esistente per ricrearne una pulita
        logger.info("Eliminazione collezione esistente...")
        try:
            qdrant_manager.client.delete_collection(COLLECTION_NAME)
            logger.info("Collezione eliminata con successo")
        except Exception as e:
            logger.warning(f"Errore nell'eliminazione collezione (potrebbe non esistere): {e}")
        
        # Re-crea la collezione
        logger.info("Creazione nuova collezione...")
        if not qdrant_manager.ensure_collection_exists(embedder.embedding_dim):
            logger.error("Impossibile creare la collezione")
            return False
        
        # Re-indicizza tutti i PDF nella cartella
        pdf_files = list(Path("data/raw").glob("*.pdf"))
        if not pdf_files:
            logger.error("Nessun file PDF trovato in data/raw/")
            return False
        
        logger.info(f"Trovati {len(pdf_files)} file PDF da re-indicizzare")
        
        for pdf_file in pdf_files:
            logger.info(f"Re-indicizzazione di: {pdf_file}")
            indexer.index_files([str(pdf_file)], force_recreate=False)
        
        # Verifica i risultati
        logger.info("Verifica dei risultati...")
        info = qdrant_manager.client.get_collection(COLLECTION_NAME)
        logger.info(f"Punti nel database dopo re-indicizzazione: {info.points_count}")
        
        # Prendi alcuni esempi per verificare i nuovi campi
        points = qdrant_manager.client.scroll(COLLECTION_NAME, limit=5, with_payload=True)
        
        caption_count = 0
        context_count = 0
        
        for point in points[0]:
            payload = point.payload
            metadata = payload.get('metadata', {})
            
            # Conta elementi con caption
            if (metadata.get('caption') or metadata.get('manual_caption')):
                caption_count += 1
            
            # Conta elementi con contesto
            if metadata.get('context_text'):
                context_count += 1
        
        logger.info(f"Elementi con caption: {caption_count}/5 verificati")
        logger.info(f"Elementi con contesto: {context_count}/5 verificati")
        
        # Mostra un esempio dettagliato
        if points[0]:
            example = points[0][0]
            payload = example.payload
            metadata = payload.get('metadata', {})
            content_type = payload.get('content_type', 'unknown')
            
            logger.info(f"\nEsempio dettagliato - Tipo: {content_type}")
            if content_type == 'table':
                logger.info(f"  Caption tabella: {metadata.get('caption', 'MANCANTE')}")
                logger.info(f"  Contesto: {metadata.get('context_text', 'MANCANTE')[:100]}...")
            elif content_type == 'image':
                logger.info(f"  Caption immagine: {metadata.get('manual_caption', 'MANCANTE')}")
                logger.info(f"  Contesto: {metadata.get('context_text', 'MANCANTE')[:100]}...")
        
        logger.info("Re-indicizzazione completata con successo!")
        return True
        
    except Exception as e:
        logger.error(f"Errore durante la re-indicizzazione: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = reindex_with_context()
    print(f"\nRe-indicizzazione {'RIUSCITA' if success else 'FALLITA'}!")
    sys.exit(0 if success else 1)
