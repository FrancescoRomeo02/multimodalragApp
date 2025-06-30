#!/usr/bin/env python3
"""
Script per debuggare l'estrazione di contesto - analizza il layout del PDF
"""

import sys
import logging
from pathlib import Path
import fitz

# Aggiungi il percorso del modulo principale
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.context_extractor import ContextExtractor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_pdf_layout(pdf_path: str, page_num: int = 0):
    """
    Analizza il layout di una specifica pagina del PDF per debug
    """
    logger.info(f"Analizzando layout della pagina {page_num+1} di: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        print(f"\n=== ANALISI PAGINA {page_num+1} ===")
        print(f"Dimensioni pagina: {page.rect}")
        
        # Estrai tutti i blocchi di testo con posizioni
        text_dict = page.get_text("dict")
        
        print(f"\n=== BLOCCHI DI TESTO ===")
        for i, block in enumerate(text_dict["blocks"]):
            if "lines" in block:  # Blocco di testo
                block_text = ""
                block_bbox = block["bbox"]
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                    block_text += " "
                
                if block_text.strip():
                    print(f"\nBlocco {i}:")
                    print(f"  BBox: {block_bbox}")
                    print(f"  Y-center: {(block_bbox[1] + block_bbox[3]) / 2:.1f}")
                    text_preview = block_text.strip()[:100] + "..." if len(block_text.strip()) > 100 else block_text.strip()
                    print(f"  Testo: {text_preview}")
        
        # Estrai tabelle
        print(f"\n=== TABELLE ===")
        tabs = page.find_tables()
        if tabs.tables:
            for i, table in enumerate(tabs.tables):
                print(f"\nTabella {i+1}:")
                print(f"  BBox: {table.bbox}")
                print(f"  Y-center: {(table.bbox[1] + table.bbox[3]) / 2:.1f}")
                
                # Ora testiamo l'estrazione del contesto per questa tabella
                context_extractor = ContextExtractor(context_window=500, max_distance=300)
                context_info = context_extractor.extract_table_context(table.bbox, page)
                
                print(f"  Caption estratta: {context_info.get('caption', 'NESSUNA')}")
                context_text = context_info.get('context_text', 'NESSUNO')
                if context_text and len(context_text) > 100:
                    context_text = context_text[:100] + "..."
                print(f"  Contesto estratto: {context_text}")
        else:
            print("Nessuna tabella trovata su questa pagina")
        
        # Estrai immagini
        print(f"\n=== IMMAGINI ===")
        images = page.get_images(full=True)
        if images:
            for i, img_info in enumerate(images):
                xref = img_info[0]
                img_rects = [rect for rect in page.get_image_rects(xref)]
                if img_rects:
                    image_rect = img_rects[0]
                    print(f"\nImmagine {i+1}:")
                    print(f"  BBox: [{image_rect.x0}, {image_rect.y0}, {image_rect.x1}, {image_rect.y1}]")
                    print(f"  Y-center: {(image_rect.y0 + image_rect.y1) / 2:.1f}")
                    
                    # Testiamo l'estrazione del contesto per questa immagine
                    context_extractor = ContextExtractor(context_window=500, max_distance=300)
                    context_info = context_extractor.extract_image_context(image_rect, page)
                    
                    print(f"  Caption estratta: {context_info.get('caption', 'NESSUNA')}")
                    context_text = context_info.get('context_text', 'NESSUNO')
                    if context_text and len(context_text) > 100:
                        context_text = context_text[:100] + "..."
                    print(f"  Contesto estratto: {context_text}")
        else:
            print("Nessuna immagine trovata su questa pagina")
            
    except Exception as e:
        logger.error(f"Errore durante l'analisi: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python debug_context_extraction.py <percorso_pdf> [numero_pagina]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0  # Default prima pagina
    
    if not Path(pdf_path).exists():
        logger.error(f"File PDF non trovato: {pdf_path}")
        sys.exit(1)
    
    success = debug_pdf_layout(pdf_path, page_num)
    sys.exit(0 if success else 1)
