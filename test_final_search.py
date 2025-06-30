#!/usr/bin/env python3
"""
Test finale per verificare la ricerca con dati del parser unificato.
"""

import sys
import os
import logging

# Aggiungi il percorso dell'app al Python path
sys.path.append('/Users/fraromeo/Documents/02_Areas/University/LM/LM_24-25/SEM2/AD/multimodalrag')

from app.pipeline.retriever import DocumentRetriever
from app.utils.embedder import AdvancedEmbedder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_search_with_unified_parser():
    """Test ricerca con dati del parser unificato."""
    
    try:
        logger.info("Avvio test ricerca con parser unificato...")
        
        # Inizializza componenti
        embedder = AdvancedEmbedder()
        retriever = DocumentRetriever(embedder)
        
        # Test queries che dovrebbero beneficiare delle funzionalità AI
        queries = [
            "gatto nero",
            "cat on roof", 
            "black cat",
            "animale sul tetto",
            "immagine con testo"
        ]
        
        for query in queries:
            logger.info(f"\n🔍 Query: '{query}'")
            results = retriever.search(query, limit=3)
            
            if results:
                for i, result in enumerate(results):
                    logger.info(f"   Risultato {i+1} (score: {result.score:.3f}):")
                    metadata = result.metadata
                    
                    if metadata.get('type') == 'image':
                        logger.info(f"     - AI Caption: {metadata.get('ai_caption', 'N/A')[:100]}...")
                        logger.info(f"     - OCR Text: {metadata.get('ocr_text', 'N/A')[:50]}...")
                        logger.info(f"     - Objects: {metadata.get('detected_objects', [])}")
                    
                    logger.info(f"     - Content: {result.content[:100]}...")
            else:
                logger.info("   Nessun risultato trovato")
        
        logger.info("\n✅ Test ricerca completato con successo!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Errore nel test ricerca: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_search_with_unified_parser()
    sys.exit(0 if success else 1)
