import fitz  
import base64
import logging
from typing import Tuple, List, Dict, Any
from io import BytesIO
from PIL import Image as PILImage
import os
import pandas as pd
import numpy as np
from src.utils.context_extractor import ContextExtractor
from src.utils.image_info import get_comprehensive_image_info
from src.utils.table_info import enhance_table_with_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_valid_table(df: pd.DataFrame) -> bool:
    """
    Filtro avanzato per validazione tabelle con criteri migliorati:
    - Dimensioni minime: almeno 2 righe e 2 colonne
    - Qualità dei dati: meno del 70% di celle vuote
    - Contenuto significativo: almeno il 30% delle celle con testo valido
    - Struttura coerente: varianza nella lunghezza del contenuto
    - Header detection: prima riga con caratteristiche di intestazione
    """
    rows, cols = df.shape
    
    
    total_cells = rows * cols
    empty_cells = df.isna().values.sum()
    empty_ratio = empty_cells / total_cells
    
    # Soglia più rigorosa per celle vuote
    if empty_ratio > 0.7:
        logger.debug(f"Tabella scartata: troppe celle vuote ({empty_ratio:.2%})")
        return False
    
    # Conta celle con contenuto significativo (non solo spazi o caratteri singoli)
    meaningful_content = 0
    for col in df.columns:
        for value in df[col].dropna():
            if isinstance(value, str) and len(str(value).strip()) > 1:
                meaningful_content += 1
    
    meaningful_ratio = meaningful_content / (total_cells - empty_cells) if (total_cells - empty_cells) > 0 else 0
    if meaningful_ratio < 0.3:
        logger.debug(f"Tabella scartata: contenuto poco significativo ({meaningful_ratio:.2%})")
        return False
    
    # Verifica presenza di header (prima riga diversa dalle altre)
    if rows >= 3:
        first_row = df.iloc[0].astype(str)
        second_row = df.iloc[1].astype(str)
        
        # Header spesso contiene testo più breve e descrittivo
        first_row_lengths = [len(str(x).strip()) for x in first_row if pd.notna(x)]
        second_row_lengths = [len(str(x).strip()) for x in second_row if pd.notna(x)]
        
        avg_first = np.mean(first_row_lengths) if first_row_lengths else 0
        avg_second = np.mean(second_row_lengths) if second_row_lengths else 0
        
        # Se la prima riga ha contenuto troppo simile alle altre, potrebbe non essere una tabella
        if abs(avg_first - avg_second) < 2 and avg_first < 3:
            logger.debug("Tabella scartata: struttura header non chiara")
            return False
    
    # Verifica varianza nel contenuto (tabelle reali hanno dati diversificati)
    content_variance_score = 0
    text_only = True
    for col in df.columns:
        col_values = df[col].dropna().astype(str)
        # Verifica se ci sono valori numerici nella colonna
        if any(col_values.str.replace('.', '', 1).str.isdigit()):
            text_only = False
        if len(col_values) > 1:
            unique_values = col_values.nunique()
            variance_ratio = unique_values / len(col_values)
            content_variance_score += variance_ratio

    avg_variance = content_variance_score / cols if cols > 0 else 0
    # Se la tabella contiene solo testo, salta il controllo di varianza
    if not text_only and avg_variance < 0.1:  # Troppo poco variabile
        logger.debug(f"Tabella scartata: contenuto troppo uniforme (variance: {avg_variance:.2f})")
        return False
    
    
    logger.debug(f"Tabella valida: {rows}x{cols}, vuote: {empty_ratio:.2%}, significative: {meaningful_ratio:.2%}")

    return True

def is_valid_image(width: int, height: int, image_data: bytes) -> bool:
    """
    Filtro per immagini valide basato su dimensioni e qualità:
    - Dimensioni minime: 100x100 pixel
    - Area minima: 15.000 pixel
    - Dimensioni massime ragionevoli: 5000x5000 pixel
    - Dimensione file minima: 1KB per evitare placeholder/icone
    - Rapporto di aspetto ragionevole (non troppo allungate)
    """
    # Controlli dimensioni di base
    if width < 50 or height < 50:
        logger.debug(f"Immagine scartata: dimensioni troppo piccole ({width}x{height})")
        return False
    
    # Controllo area minima
    area = width * height
    if area < 2500:  # ~50x50 pixel
        logger.debug(f"Immagine scartata: area troppo piccola ({area} pixel)")
        return False
    
    # Controllo dimensioni massime
    if width > 5000 or height > 5000:
        logger.debug(f"Immagine scartata: dimensioni troppo grandi ({width}x{height})")
        return False
    
    # Controllo dimensione file
    if len(image_data) < 1024:  # 1KB minimo
        logger.debug(f"Immagine scartata: file troppo piccolo ({len(image_data)} bytes)")
        return False
    
    # Controllo rapporto di aspetto (evita immagini troppo allungate)
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 10:  # Rapporto massimo 10:1
        logger.debug(f"Immagine scartata: rapporto di aspetto troppo estremo ({aspect_ratio:.2f})")
        return False
    
    return True


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
        if not tabs or len(tabs.tables) == 0:
            return tables
            
        for i, table in enumerate(tabs.tables):
            try:
                # Verifica preliminare del bounding box
                if not hasattr(table, 'bbox') or not table.bbox:
                    continue
                    
                df = table.to_pandas()
                
                # Converti tutti i valori a stringa e pulisci
                df = df.map(lambda x: str(x).strip() if pd.notna(x) else x)
                df = df.replace(r'^\s*$', np.nan, regex=True)
                df = df.dropna(how='all').dropna(how='all', axis=1)
                
                # Validazione tabella con informazioni dettagliate
                if not is_valid_table(df):
                    rows, cols = df.shape
                    empty_ratio = df.isna().values.sum() / (rows * cols) if rows * cols > 0 else 1
                    logger.info(f"Tabella {i+1} pagina {page.number if page.number is not None else 0 + 1} scartata")
                    logger.debug(df)
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
                
                # Log informativo per tabelle accettate
                rows, cols = df.shape
                empty_ratio = df.isna().values.sum() / (rows * cols)
                logger.info(f"Tabella {i+1} pagina {page.number+1} accettata: {rows}x{cols} celle, {empty_ratio:.1%} vuote")
                
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
                    
                    table_element = {
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
                            "context_text": table_context.get("context_text"),
                            "table_summary": None  # Placeholder per il riassunto AI
                        }
                    }
                    # Arricchisci la tabella con il riassunto AI
                    table_element = enhance_table_with_summary(table_element)
                    table_elements.append(table_element)


                except Exception as e:
                    logger.warning(f"Errore nell'elaborazione della tabella: {str(e)}")
            
            # Estrazione immagini con metadati standardizzati, contesto manuale e analisi AI
            for img_index, img_info in enumerate(page.get_images(full=True)):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    
                    # Controllo validità dell'immagine con criteri migliorati
                    if not base_image["image"]:
                        logger.debug(f"Immagine {img_index+1} pagina {page_num+1}: dati immagine mancanti")
                        continue
                        
                    if not is_valid_image(base_image["width"], base_image["height"], base_image["image"]):
                        logger.debug(f"Immagine {img_index+1} pagina {page_num+1} scartata per dimensioni/qualità")
                        continue
                    
                    # Log informativo per immagini accettate
                    logger.info(f"Immagine {img_index+1} pagina {page_num+1} accettata: {base_image['width']}x{base_image['height']} pixel, {len(base_image['image'])/1024:.1f}KB")
                    
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