import fitz  
import base64
import logging
from typing import Tuple, List, Dict, Any
from io import BytesIO
from PIL import Image as PILImage
import os
from app.core.models import ColorSpace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_pdf_elements(pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parsing PDF con metadati standardizzati per compatibilità con Qdrant
    """
    logger.info(f"Avvio parsing avanzato del PDF: {os.path.basename(pdf_path)}")

    text_elements = []
    image_elements = []
    
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
                        # CAMPI STANDARDIZZATI per il retriever
                        "type": "text",
                        "source": filename,
                        "page": page_num + 1,
                        
                        # Campi aggiuntivi per compatibilità
                        "content_type": "text",
                        }
                    })
            
            # Estrazione immagini con metadati standardizzati
            for img_index, img_info in enumerate(page.get_images(full=True)):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    
                    # Filtra immagini troppo piccole o corrotte
                    if (base_image["width"] < 50 or base_image["height"] < 50 or 
                        not base_image["image"]):
                        logger.debug(f"Saltata immagine troppo piccola: {base_image['width']}x{base_image['height']}")
                        continue
                    
                    # Processa l'immagine
                    with BytesIO(base_image["image"]) as img_buffer:
                        img = PILImage.open(img_buffer)
                        img_format = img.format or "PNG"
                        
                        # Ricodifica ottimizzata
                        with BytesIO() as output_buffer:
                            img.save(output_buffer, format=img_format, optimize=True, quality=85)
                            optimized_image = output_buffer.getvalue()
                        
                        # Costruisci metadati STANDARDIZZATI
                        image_metadata = {
                            # CAMPI STANDARDIZZATI per il retriever
                            "type": "image",
                            "source": filename,
                            "page": page_num + 1,
                            
                            # Campi aggiuntivi
                            "content_type": "image",
                            "image_base64": base64.b64encode(optimized_image).decode("utf-8"),
                        }
                        
                        # Descrizione per l'embedding testuale
                        description = f"Immagine {img_index+1} a pagina {page_num+1} - {img_format.upper()} ({base_image['width']}x{base_image['height']})"
                        
                        image_elements.append({
                            "image_base64": base64.b64encode(optimized_image).decode("utf-8"),
                            "metadata": image_metadata,
                            "page_content": description
                        })
                        
                        logger.debug(f"Processata immagine {img_index+1} pagina {page_num+1}: {img_format} {base_image['width']}x{base_image['height']}")
                        
                except Exception as img_e:
                    logger.error(f"Errore processamento immagine {img_index} pagina {page_num+1}: {str(img_e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Errore durante il parsing del PDF: {str(e)}")
        raise RuntimeError(f"PDF parsing failed: {str(e)}") from e
    
    logger.info(f"Estrazione completata: {len(text_elements)} testi, {len(image_elements)} immagini valide")
    return text_elements, image_elements