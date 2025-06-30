#!/usr/bin/env python3
"""
Script per testare l'estrazione di contesto da PDF
"""

import sys
import logging
from pathlib import Path

# Aggiungi il percorso del modulo principale
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.pdf_parser import parse_pdf_elements
from app.utils.context_extractor import ContextExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_context_extraction(pdf_path: str):
    """
    Testa l'estrazione di contesto da un PDF
    """
    logger.info(f"Testando estrazione contesto da: {pdf_path}")
    
    try:
        # Parsing del PDF con estrazione contesto
        text_elements, image_elements, table_elements = parse_pdf_elements(pdf_path)
        
        logger.info(f"Elementi estratti: {len(text_elements)} testi, {len(image_elements)} immagini, {len(table_elements)} tabelle")
        
        # Mostra alcuni esempi di tabelle con contesto
        logger.info("\n=== TABELLE CON CONTESTO ===")
        for i, table in enumerate(table_elements[:3]):  # Primi 3
            metadata = table["metadata"]
            logger.info(f"\nTabella {i+1}:")
            logger.info(f"  Pagina: {metadata['page']}")
            logger.info(f"  Caption: {metadata.get('caption', 'N/A')}")
            logger.info(f"  Contesto: {metadata.get('context_text', 'N/A')}")
            logger.info(f"  Forma: {metadata.get('table_shape', 'N/A')}")
            
            # Mostra parte del contenuto arricchito
            content = table["table_markdown"]
            preview = content[:200] + "..." if len(content) > 200 else content
            logger.info(f"  Contenuto arricchito: {preview}")
        
        # Mostra alcuni esempi di immagini con contesto
        logger.info("\n=== IMMAGINI CON CONTESTO ===")
        for i, image in enumerate(image_elements[:3]):  # Primi 3
            metadata = image["metadata"]
            logger.info(f"\nImmagine {i+1}:")
            logger.info(f"  Pagina: {metadata['page']}")
            logger.info(f"  Caption manuale: {metadata.get('manual_caption', 'N/A')}")
            logger.info(f"  Contesto: {metadata.get('context_text', 'N/A')}")
            
            # Mostra parte della descrizione arricchita
            content = image["page_content"]
            preview = content[:200] + "..." if len(content) > 200 else content
            logger.info(f"  Descrizione arricchita: {preview}")
            
        logger.info(f"\nTest completato con successo!")
        
    except Exception as e:
        logger.error(f"Errore durante il test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python test_context_extraction.py <percorso_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        logger.error(f"File PDF non trovato: {pdf_path}")
        sys.exit(1)
    
    success = test_context_extraction(pdf_path)
    sys.exit(0 if success else 1)
