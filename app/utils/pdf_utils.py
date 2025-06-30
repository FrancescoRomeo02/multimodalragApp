from app.utils.image_info import get_comprehensive_image_info
from app.core.models import TableData, TextElement, ImageElement, TableElement
from app.utils.embedder import AdvancedEmbedder
import os
import logging
import uuid
from typing import List, Dict, Any, Tuple, Optional
from pydantic import ValidationError
from qdrant_client.http import models as qdrant_models
from datetime import datetime
import fitz  # PyMuPDF
import base64
import logging
from typing import List, Tuple, Dict, Any
from io import BytesIO
from PIL import Image as PILImage
import os
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_tables_from_page(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Estrae tabelle da una pagina PDF usando PyMuPDF.
    
    Returns:
        Lista di dizionari con dati tabellari (cells, headers, shape), markdown e bounding box.
    """
    tables = []
    try:
        tabs = page.find_tables()
        
        if not tabs.tables:
            return tables
        
        for i, table in enumerate(tabs.tables):
            try:
                df = table.to_pandas()
                
                # Pulizia dati
                df = df.replace(r'^\s*$', np.nan, regex=True)
                df = df.dropna(how='all').dropna(how='all', axis=1)
                if df.empty:
                    continue
                
                table_md = df.to_markdown(index=False)
                table_data = {
                    "cells": df.values.tolist(),
                    "headers": df.columns.tolist(),
                    "shape": df.shape
                }
                
                tables.append({
                    "table_data": table_data,
                    "table_markdown": table_md,
                    "bbox": table.bbox
                })
                
                logger.debug(f"Trovata tabella {i+1} pagina {page.number+1} shape {df.shape}")
                
            except Exception as e:
                logger.warning(f"Errore estrazione tabella {i+1} pagina {page.number+1}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Errore estrazione tabelle pagina {page.number+1}: {str(e)}")
    
    return tables


def parse_pdf_elements(pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parsing PDF per estrarre elementi testuali, immagini e tabelle con metadati standardizzati.
    
    Args:
        pdf_path: Percorso al file PDF
    
    Returns:
        Tuple con tre liste di dizionari: (testo, immagini, tabelle)
    """
    logger.info(f"Parsing PDF: {os.path.basename(pdf_path)}")
    
    text_elements = []
    image_elements = []
    table_elements = []
    
    try:
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Testo
            text = page.get_text("text").strip()
            if text:
                text_elements.append({
                    "text": text,
                    "metadata": {
                        "type": "text",
                        "source": filename,
                        "page": page_num + 1,
                        "content_type": "text"
                    }
                })
            
            # Tabelle
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
                    logger.warning(f"Errore elaborazione tabella pagina {page_num+1}: {str(e)}")
            
            # Immagini
            for img_index, img_info in enumerate(page.get_images(full=True)):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)

                    if base_image["width"] < 50 or base_image["height"] < 50 or not base_image["image"]:
                        continue

                    with BytesIO(base_image["image"]) as img_buffer:
                        img = PILImage.open(img_buffer)
                        img_format = img.format or "PNG"

                        with BytesIO() as output_buffer:
                            img.save(output_buffer, format=img_format, optimize=True, quality=85)
                            optimized_image = output_buffer.getvalue()

                    image_base64 = base64.b64encode(optimized_image).decode("utf-8")

                    # Ottieni info dettagliate sull'immagine
                    image_info = get_comprehensive_image_info(image_base64)
                    caption_parts = [
                            f"Image Description: {image_info['caption']}",
                            f"Image Text: {image_info['ocr_text']}" if image_info["ocr_text"].strip() else None,
                            f"Detected Image's Objects: {', '.join(image_info['detected_objects'])}" if image_info["detected_objects"] else None
                    ]
                    image_caption = "\n".join([p for p in caption_parts if p])
                    image_metadata = {
                    "type": "image",
                    "source": filename,
                    "page": page_num + 1,
                    "content_type": "image",
                    "image_caption": image_caption
                }

                    image_elements.append({
                    "image_base64": image_base64,
                    "metadata": image_metadata,
                    "page_content": image_caption  # usato per il retrieval semantico
                    })
                except Exception as e:
                    logger.warning(f"Errore elaborazione immagine {img_index+1} pagina {page_num+1}: {str(e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Errore parsing PDF: {str(e)}")
        raise RuntimeError(f"Parsing fallito: {str(e)}") from e
    
    logger.info(f"Parsing completato: {len(text_elements)} testi, {len(image_elements)} immagini, {len(table_elements)} tabelle")
    return text_elements, image_elements, table_elements