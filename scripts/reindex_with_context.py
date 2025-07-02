#!/usr/bin/env python3
"""
Script per re-indicizzare i documenti con il nuovo sistema di estrazione contesto
"""

import sys
import logging
from pathlib import Path

# Aggiungi il percorso del modulo principale
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.indexer_service import DocumentIndexer
from src.utils.embedder import get_multimodal_embedding_model
from src.utils.qdrant_utils import qdrant_manager
from src.config import COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reindex_with_context(use_semantic_chunking: bool = True):
    """
    Re-indicizza tutti i documenti nella cartella data/raw con chunking semantico (default)
    che include chunking semantico e estrazione contesto
    """
    chunk_type = "semantico" if use_semantic_chunking else "classico"
    logger.info(f"Inizio re-indicizzazione con chunking {chunk_type}...")
    
    try:
        # Inizializza i servizi con chunking semantico come default
        embedder = get_multimodal_embedding_model()
        indexer = DocumentIndexer(embedder=embedder, use_semantic_chunking=use_semantic_chunking)
        
        # Mostra tipo di chunking attivo
        actual_chunking = "semantico" if indexer.semantic_chunker else "classico (fallback)"
        logger.info(f"Indexer inizializzato con chunking: {actual_chunking}")
        
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
        semantic_chunk_count = 0
        
        for point in points[0]:
            payload = point.payload if point.payload else {}
            metadata = payload.get('metadata', {})
            
            # Conta elementi con caption
            if (metadata.get('caption') or metadata.get('manual_caption')):
                caption_count += 1
            
            # Conta elementi con contesto
            if metadata.get('context_text'):
                context_count += 1
            
            # Conta chunk semantici
            if metadata.get('chunk_type') == 'semantic':
                semantic_chunk_count += 1
        
        logger.info(f"Elementi con caption: {caption_count}/5 verificati")
        logger.info(f"Elementi con contesto: {context_count}/5 verificati")
        logger.info(f"Chunk semantici: {semantic_chunk_count}/5 verificati")
        
        # Mostra un esempio dettagliato
        if points[0]:
            example = points[0][0]
            payload = example.payload if example.payload else {}
            metadata = payload.get('metadata', {})
            content_type = payload.get('content_type', 'unknown')
            
            logger.info(f"\nEsempio dettagliato - Tipo: {content_type}")
            
            # Informazioni chunking
            if metadata.get('chunk_type'):
                logger.info(f"  Tipo chunk: {metadata.get('chunk_type')}")
                logger.info(f"  Chunk ID: {metadata.get('chunk_id', 'N/A')}")
                logger.info(f"  Lunghezza chunk: {metadata.get('chunk_length', 'N/A')}")
            
            if content_type == 'table':
                logger.info(f"  Caption tabella: {metadata.get('caption', 'MANCANTE')}")
                logger.info(f"  Contesto: {metadata.get('context_text', 'MANCANTE')[:100]}...")
            elif content_type == 'image':
                logger.info(f"  Caption immagine: {metadata.get('manual_caption', 'MANCANTE')}")
                logger.info(f"  Contesto: {metadata.get('context_text', 'MANCANTE')[:100]}...")
            elif content_type == 'text':
                logger.info(f"  Testo: {payload.get('page_content', 'MANCANTE')[:100]}...")
        
        logger.info("Re-indicizzazione completata con successo!")
        return True
        
    except Exception as e:
        logger.error(f"Errore durante la re-indicizzazione: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-indicizza documenti con chunking semantico (default)")
    parser.add_argument("--classic", action="store_true", 
                       help="Usa chunking classico invece del chunking semantico (default)")
    
    args = parser.parse_args()
    
    # Default è chunking semantico, --classic lo disabilita
    use_semantic = not args.classic
    success = reindex_with_context(use_semantic_chunking=use_semantic)
    
    chunk_type = "semantico" if use_semantic else "classico"
    print(f"\nRe-indicizzazione con chunking {chunk_type} {'RIUSCITA' if success else 'FALLITA'}!")
    
    if success and use_semantic:
        print("🧠 Chunking semantico attivo: segmentazione intelligente dei documenti")
    elif success:
        print("📝 Chunking classico attivo: segmentazione per lunghezza")
        
    sys.exit(0 if success else 1)
