import fitz  
import base64
import logging
from typing import Tuple, List, Dict, Any
from io import BytesIO
from PIL import Image as PILImage
import os
import pandas as pd
import numpy as np

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
    Parsing PDF con metadati standardizzati per compatibilità con Qdrant
    """
    logger.info(f"Avvio parsing avanzato del PDF: {os.path.basename(pdf_path)}")

    text_elements = []
    image_elements = []
    table_elements = []
    
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
            
            # Estrazione tabelle
            tables = extract_tables_from_page(page)
            for table in tables:
                try:
                    table_elements.append({
                        "table_data": table["table_data"],
                        "table_markdown": table["table_markdown"],
                        "metadata": {
                            "type": "table",
                            "source": filename,
                            "page": page_num + 1,
                            "content_type": "table",
                            "bbox": table["bbox"],
                            "table_shape": table["table_data"]["shape"]
                        }
                    })
                except Exception as e:
                    logger.warning(f"Errore nell'elaborazione della tabella: {str(e)}")
            
            # Estrazione immagini con metadati standardizzati
            for img_index, img_info in enumerate(page.get_images(full=True)):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    
                    if (base_image["width"] < 50 or base_image["height"] < 50 or 
                        not base_image["image"]):
                        continue
                    
                    with BytesIO(base_image["image"]) as img_buffer:
                        img = PILImage.open(img_buffer)
                        img_format = img.format or "PNG"
                        
                        with BytesIO() as output_buffer:
                            img.save(output_buffer, format=img_format, optimize=True, quality=85)
                            optimized_image = output_buffer.getvalue()
                        
                        image_metadata = {
                            "type": "image",
                            "source": filename,
                            "page": page_num + 1,
                            "content_type": "image",
                            "image_base64": base64.b64encode(optimized_image).decode("utf-8"),
                        }
                        
                        description = f"Immagine {img_index+1} a pagina {page_num+1} - {img_format.upper()} ({base_image['width']}x{base_image['height']})"
                        
                        image_elements.append({
                            "image_base64": base64.b64encode(optimized_image).decode("utf-8"),
                            "metadata": image_metadata,
                            "page_content": description
                        })
                        
                except Exception as img_e:
                    logger.error(f"Errore processamento immagine {img_index} pagina {page_num+1}: {str(img_e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Errore durante il parsing del PDF: {str(e)}")
        raise RuntimeError(f"PDF parsing failed: {str(e)}") from e
    
    logger.info(f"Estrazione completata: {len(text_elements)} testi, {len(image_elements)} immagini, {len(table_elements)} tabelle")
    return text_elements, image_elements, table_elements