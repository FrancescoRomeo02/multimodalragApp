#!/usr/bin/env python3
"""
Test rapido per verificare che l'indexer funzioni con il parser unificato.
"""

import sys
import os
import logging

# Aggiungi il percorso dell'app al Python path
sys.path.append('/Users/fraromeo/Documents/02_Areas/University/LM/LM_24-25/SEM2/AD/multimodalrag')

from app.pipeline.indexer_service import DocumentIndexer
from app.utils.embedder import AdvancedEmbedder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_indexer_integration():
    """Test integrazione indexer con parser unificato."""
    
    pdf_path = "/Users/fraromeo/Documents/02_Areas/University/LM/LM_24-25/SEM2/AD/multimodalrag/data/raw/gatto.pdf"
    
    if not os.path.exists(pdf_path):
        logger.error(f"File PDF non trovato: {pdf_path}")
        return False
    
    try:
        logger.info("Avvio test integrazione indexer + parser unificato...")
        
        # Inizializza componenti
        embedder = AdvancedEmbedder()
        indexer = DocumentIndexer(embedder)
        
        # Test indicizzazione
        success = indexer.index_files([pdf_path], force_recreate=True)
        
        if success:
            logger.info("✅ Test integrazione completato con successo!")
            return True
        else:
            logger.error("❌ Test integrazione fallito")
            return False
        
    except Exception as e:
        logger.error(f"❌ Errore nel test integrazione: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_indexer_integration()
    sys.exit(0 if success else 1)
