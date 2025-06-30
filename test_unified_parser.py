#!/usr/bin/env python3
"""
Test script per verificare il parser PDF unificato.
"""

import sys
import os
import logging

# Aggiungi il percorso dell'app al Python path
sys.path.append('/Users/fraromeo/Documents/02_Areas/University/LM/LM_24-25/SEM2/AD/multimodalrag')

from app.utils.pdf_parser import parse_pdf_elements

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_unified_parser():
    """Test del parser PDF unificato."""
    
    pdf_path = "/Users/fraromeo/Documents/02_Areas/University/LM/LM_24-25/SEM2/AD/multimodalrag/data/raw/gatto.pdf"
    
    if not os.path.exists(pdf_path):
        logger.error(f"File PDF non trovato: {pdf_path}")
        return False
    
    try:
        logger.info("Avvio test del parser unificato...")
        
        # Test del parsing
        text_elements, image_elements, table_elements = parse_pdf_elements(pdf_path)
        
        logger.info(f"✅ Parsing completato con successo!")
        logger.info(f"   - Elementi testuali: {len(text_elements)}")
        logger.info(f"   - Elementi immagine: {len(image_elements)}")
        logger.info(f"   - Elementi tabella: {len(table_elements)}")
        
        # Test dettagliato delle immagini
        if image_elements:
            logger.info("\n📷 Test analisi immagini:")
            for i, img_elem in enumerate(image_elements[:2]):  # Test solo le prime 2
                metadata = img_elem['metadata']
                logger.info(f"   Immagine {i+1}:")
                logger.info(f"     - Pagina: {metadata['page']}")
                logger.info(f"     - AI Caption: {metadata.get('ai_caption', 'N/A')[:100]}...")
                logger.info(f"     - OCR Text: {metadata.get('ocr_text', 'N/A')[:50]}...")
                logger.info(f"     - Detected Objects: {len(metadata.get('detected_objects', []))}")
                logger.info(f"     - Manual Caption: {metadata.get('manual_caption', 'N/A')}")
        
        # Test dettagliato delle tabelle
        if table_elements:
            logger.info("\n📊 Test analisi tabelle:")
            for i, table_elem in enumerate(table_elements[:2]):  # Test solo le prime 2
                metadata = table_elem['metadata']
                logger.info(f"   Tabella {i+1}:")
                logger.info(f"     - Pagina: {metadata['page']}")
                logger.info(f"     - Shape: {metadata.get('table_shape', 'N/A')}")
                logger.info(f"     - Caption: {metadata.get('caption', 'N/A')}")
                logger.info(f"     - Context Text: {metadata.get('context_text', 'N/A')[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Errore nel test del parser: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unified_parser()
    sys.exit(0 if success else 1)
