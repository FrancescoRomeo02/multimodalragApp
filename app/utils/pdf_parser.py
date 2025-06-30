import fitz  
import base64
import logging
from typing import Tuple, List, Dict, Any
from io import BytesIO
from PIL import Image as PILImage
import os
import pandas as pd
import numpy as np
from app.utils.context_extractor import ContextExtractor
from app.utils.image_info import get_comprehensive_image_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_tables_from_page(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Estrae tabelle da una pagina PDF usando PyMuPDF
    
    Args:
        page: La pagina PDF da analizzare
        
    Returns:
        Lista di dizionari contenenti i dati delle tabelle
    """
    tables = []
    try:
        # Metodo principale: Rilevamento tabelle di PyMuPDF
        tabs = page.find_tables()
        
        if len(tabs.tables) == 0:
            return tables
            
        for i, table in enumerate(tabs.tables):
            try:
                # Estrai i dati della tabella
                df = table.to_pandas()
                
                # Pulisci i dati della tabella
                df = df.replace(r'^\s*$', np.nan, regex=True)
                df = df.dropna(how='all').dropna(how='all', axis=1)
                
                if df.empty:
                    continue
                
                # Converti la tabella in formato markdown
                table_md = df.to_markdown(index=False)
                
                # Crea una rappresentazione strutturata della tabella
                table_data = {
                    "cells": df.values.tolist(),
                    "headers": df.columns.tolist(),
                    "shape": df.shape
                }
                
                tables.append({
                    "table_data": table_data,
                    "table_markdown": table_md,
                    "bbox": table.bbox  # Rettangolo che contiene la tabella
                })
                
                logger.debug(f"Trovata tabella {i+1} a pagina {page.number+1} con {df.shape} celle")
                
            except Exception as table_e:
                logger.warning(f"Errore nell'estrazione della tabella {i+1}: {str(table_e)}")
                continue
                
    except Exception as e:
        logger.error(f"Errore generale nell'estrazione delle tabelle: {str(e)}")
    
    return tables

def parse_pdf_elements(pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parser PDF unificato con funzionalità avanzate complete.
    
    FUNZIONALITÀ INTEGRATE:
    - Estrazione testo, immagini e tabelle standardizzata per Qdrant
    - Analisi AI delle immagini: caption BLIP, OCR Tesseract, object detection YOLO
    - Estrazione contesto manuale per tabelle e immagini
    - Combinazione contesto AI + manuale per ricerca semantica potenziata
    - Metadati completi e strutturati per ogni elemento
    
    Args:
        pdf_path: Percorso al file PDF da processare
        
    Returns:
        Tuple di tre liste:
        - text_elements: Lista di elementi testuali con metadati
        - image_elements: Lista di immagini con analisi AI e contesto
        - table_elements: Lista di tabelle con contesto e markup
        
    Note:
        Questo parser sostituisce e unifica le funzionalità precedentemente
        distribuite tra pdf_parser.py e pdf_utils.py, fornendo il meglio
        di entrambe le implementazioni in un'unica soluzione.
    """
    logger.info(f"Avvio parsing avanzato del PDF: {os.path.basename(pdf_path)}")

    text_elements = []
    image_elements = []
    table_elements = []
    
    # Inizializza l'estrattore di contesto con parametri più ampi
    context_extractor = ContextExtractor(context_window=500, max_distance=300)
    
    try:
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_bbox = page.rect
            
            # Estrazione testo con metadati STANDARDIZZATI
            text = page.get_text("text").strip()
            if text:
                text_elements.append({
                    "text": text,
                    "metadata": {
                        "type": "text",
                        "source": filename,
                        "page": page_num + 1,
                        "content_type": "text",
                    }
                })
            
            # Estrazione tabelle con contesto
            tables = extract_tables_from_page(page)
            for table in tables:
                try:
                    # Estrai contesto per la tabella
                    table_context = context_extractor.extract_table_context(table["bbox"], page)
                    
                    # Combina il markdown della tabella con il contesto per la ricerca
                    enhanced_table_content = context_extractor.enhance_text_with_context(
                        table["table_markdown"], 
                        table_context
                    )
                    
                    table_elements.append({
                        "table_data": table["table_data"],
                        "table_markdown": enhanced_table_content,  # Contenuto arricchito per la ricerca
                        "table_markdown_raw": table["table_markdown"],  # Markdown originale
                        "metadata": {
                            "type": "table",
                            "source": filename,
                            "page": page_num + 1,
                            "content_type": "table",
                            "bbox": table["bbox"],
                            "table_shape": table["table_data"]["shape"],
                            "caption": table_context.get("caption"),
                            "context_text": table_context.get("context_text")
                        }
                    })
                except Exception as e:
                    logger.warning(f"Errore nell'elaborazione della tabella: {str(e)}")
            
            # Estrazione immagini con metadati standardizzati, contesto manuale e analisi AI
            for img_index, img_info in enumerate(page.get_images(full=True)):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    
                    if (base_image["width"] < 50 or base_image["height"] < 50 or 
                        not base_image["image"]):
                        continue
                    
                    # Ottieni il rettangolo dell'immagine per estrarre il contesto
                    img_rects = [rect for rect in page.get_image_rects(xref)]
                    image_rect = img_rects[0] if img_rects else page.rect
                    
                    # Estrai contesto manuale per l'immagine
                    image_context = context_extractor.extract_image_context(image_rect, page)
                    
                    with BytesIO(base_image["image"]) as img_buffer:
                        img = PILImage.open(img_buffer)
                        img_format = img.format or "PNG"
                        
                        with BytesIO() as output_buffer:
                            img.save(output_buffer, format=img_format, optimize=True, quality=85)
                            optimized_image = output_buffer.getvalue()
                    
                    # Ottieni informazioni complete dall'immagine usando AI
                    image_base64 = base64.b64encode(optimized_image).decode("utf-8")
                    image_info_ai = get_comprehensive_image_info(image_base64)
                    
                    # Combina caption AI, OCR, e contesto manuale
                    caption_parts = [
                        f"Image's Description: {image_info_ai['caption']}",
                        f"Image's Text: {image_info_ai['ocr_text']}" if image_info_ai["ocr_text"].strip() else None,
                        f"Detected Image's Objects: {', '.join(image_info_ai['detected_objects'])}" if image_info_ai["detected_objects"] else None,
                        f"Image's Caption: {image_context.get('caption')}" if image_context.get('caption') else None
                    ]
                    comprehensive_caption = "\n".join([p for p in caption_parts if p])
                    
                    # Combina caption AI con contesto manuale per la ricerca
                    enhanced_description = context_extractor.enhance_text_with_context(
                        comprehensive_caption,
                        image_context
                    )
                    
                    logger.debug(f"Immagine {img_index+1} pagina {page_num+1} - Caption completa: {comprehensive_caption}")
                    
                    image_metadata = {
                        "type": "image",
                        "source": filename,
                        "page": page_num + 1,
                        "content_type": "image",
                        "image_caption": comprehensive_caption,  # Caption completa per compatibilità
                        "ai_caption": image_info_ai['caption'],
                        "ocr_text": image_info_ai['ocr_text'],
                        "detected_objects": image_info_ai['detected_objects'],
                        "manual_caption": image_context.get("caption"),
                        "context_text": image_context.get("context_text")
                    }
                    
                    image_elements.append({
                        "image_base64": image_base64,
                        "metadata": image_metadata,
                        "page_content": enhanced_description  # Descrizione arricchita per la ricerca
                    })
                        
                except Exception as img_e:
                    logger.error(f"Errore processamento immagine {img_index} pagina {page_num+1}: {str(img_e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Errore durante il parsing del PDF: {str(e)}")
        raise RuntimeError(f"PDF parsing failed: {str(e)}") from e
    
    logger.info(f"Estrazione completata: {len(text_elements)} testi, {len(image_elements)} immagini, {len(table_elements)} tabelle")
    return text_elements, image_elements, table_elements