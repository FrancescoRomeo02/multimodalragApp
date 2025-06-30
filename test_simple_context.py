#!/usr/bin/env python3
"""
Script per testare la funzionalità di estrazione contesto con un PDF di esempio
"""

import sys
import logging
from pathlib import Path
import requests
import fitz

# Aggiungi il percorso del modulo principale
sys.path.append(str(Path(__file__).parent))

from app.utils.context_extractor import ContextExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_pdf():
    """
    Crea un PDF di test con testo e una tabella semplice
    """
    test_pdf_path = "test_sample.pdf"
    
    # Crea un documento PDF semplice con PyMuPDF
    doc = fitz.open()  # Nuovo documento vuoto
    page = doc.new_page()  # Nuova pagina
    
    # Aggiungi del testo prima della tabella
    text_before = "Questo è il paragrafo che precede la tabella."
    page.insert_text((50, 100), text_before, fontsize=12)
    
    # Caption della tabella
    caption = "Tabella 1: Risultati sperimentali del modello"
    page.insert_text((50, 130), caption, fontsize=11, color=(0, 0, 0))
    
    # Simula una tabella con testo
    table_header = "Nome | Valore | Descrizione"
    page.insert_text((50, 160), table_header, fontsize=10)
    
    row1 = "Parametro A | 0.95 | Accuracy del modello"
    page.insert_text((50, 180), row1, fontsize=10)
    
    row2 = "Parametro B | 0.87 | Precision del modello"  
    page.insert_text((50, 200), row2, fontsize=10)
    
    # Testo dopo la tabella
    text_after = "Come si può vedere dalla tabella, i risultati sono eccellenti."
    page.insert_text((50, 230), text_after, fontsize=12)
    
    # Aggiungi un'immagine simulata (rettangolo)
    rect = fitz.Rect(50, 270, 200, 320)
    page.draw_rect(rect, color=(0, 0, 1), width=2)
    
    # Caption dell'immagine
    img_caption = "Figura 1: Architettura del sistema proposto"
    page.insert_text((50, 340), img_caption, fontsize=11)
    
    doc.save(test_pdf_path)
    doc.close()
    
    logger.info(f"PDF di test creato: {test_pdf_path}")
    return test_pdf_path

def test_context_extraction_simple():
    """
    Testa l'estrazione di contesto con un PDF semplice
    """
    logger.info("Creando PDF di test...")
    pdf_path = create_test_pdf()
    
    try:
        logger.info(f"Testando estrazione contesto da: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        
        # Inizializza l'estrattore con parametri ampi
        extractor = ContextExtractor(context_window=500, max_distance=300)
        
        # Estrai blocchi di testo
        text_blocks = extractor.get_text_blocks_with_position(page)
        
        logger.info(f"Trovati {len(text_blocks)} blocchi di testo:")
        for i, block in enumerate(text_blocks):
            logger.info(f"  Blocco {i}: {block['text'][:50]}... (y={block['y_center']:.1f})")
        
        # Simula una tabella nella zona centrale
        fake_table_bbox = [45, 155, 400, 205]  # Area che copre la "tabella"
        
        logger.info(f"\nTestando estrazione contesto per tabella fittizia in bbox: {fake_table_bbox}")
        
        context_info = extractor.extract_table_context(fake_table_bbox, page)
        
        logger.info(f"Caption estratta: {context_info.get('caption', 'NESSUNA')}")
        logger.info(f"Contesto estratto: {context_info.get('context_text', 'NESSUNO')}")
        
        # Cleanup
        doc.close()
        Path(pdf_path).unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"Errore durante il test: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_context_extraction_simple()
    print(f"\nTest {'RIUSCITO' if success else 'FALLITO'}!")
    sys.exit(0 if success else 1)
