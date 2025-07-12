import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from typing import Tuple, List, Dict, Any
from src.utils.context_extractor import ContextExtractor
from src.utils.image_info import get_comprehensive_image_info
from src.utils.table_info import enhance_table_with_summary
from unstructured.partition.pdf import partition_pdf
from src.utils.pdf_validate_elemets import is_valid_image, is_valid_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    try:
        chunks = partition_pdf(
            filename=pdf_path,
            infer_table_structure=True,            # extract tables
            strategy="hi_res",                     # mandatory to infer tables

            extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
            # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

            extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

            chunking_strategy="by_title",          # or 'basic'
            max_characters=10000,                  # defaults to 500
            combine_text_under_n_chars=2000,       # defaults to 0
            new_after_n_chars=6000,
        )

        # Contatori globali per identificatori univoci
        table_counter = 0
        image_counter = 0
        page_num = 0

        tables = []
        texts = []
        images = []

        # Dictionaries to track unique elements
        unique_tables = {}
        unique_images = {}
        unique_texts = {}
        
        text_elements = []
        image_elements = []
        table_elements = []

        for chunk in chunks:
            # Ensure orig_elements exists and is not None
            if hasattr(chunk.metadata, 'orig_elements') and chunk.metadata.orig_elements:
                for el in chunk.metadata.orig_elements:
                    element_type = str(type(el))
                    
                    # Create a unique identifier for the element
                    if hasattr(el, 'text'):
                        element_content = el.text
                    elif hasattr(el, 'to_dict'):
                        element_content = str(el.to_dict())
                    else:
                        element_content = str(el)
                    
                    # Use content hash as unique identifier
                    element_id = hash(element_content)
                    
                    if "Table" in element_type:
                        if element_id not in unique_tables:
                            # Get text_as_html and page_number from the element's metadata
                            text_as_html = ''
                            page_number = None
                            if hasattr(el, 'metadata'):
                                if hasattr(el.metadata, 'text_as_html'):
                                    text_as_html = el.metadata.text_as_html
                                if hasattr(el.metadata, 'page_number'):
                                    page_number = el.metadata.page_number
                            
                            # Create a new chunk-like object for this table
                            table_chunk = type('TableChunk', (), {
                                'text': element_content,
                                'metadata': type('Metadata', (), {
                                    'text_as_html': text_as_html,
                                    'page_number': page_number,
                                    'orig_elements': [el]
                                })()
                            })()
                            unique_tables[element_id] = table_chunk
                            tables.append(table_chunk)
                            
                    elif "Image" in element_type:
                        if element_id not in unique_images:
                            # Get metadata from the original element
                            page_number = None
                            image_base64 = None
                            coordinates = None
                            if hasattr(el, 'metadata'):
                                if hasattr(el.metadata, 'page_number'):
                                    page_number = el.metadata.page_number
                                if hasattr(el.metadata, 'image_base64'):
                                    image_base64 = el.metadata.image_base64
                                if hasattr(el.metadata, 'coordinates'):
                                    coordinates = el.metadata.coordinates
                            
                            # Create a new chunk-like object for this image
                            image_chunk = type('ImageChunk', (), {
                                'text': element_content,
                                'metadata': type('Metadata', (), {
                                    'page_number': page_number,
                                    'image_base64': image_base64,
                                    'coordinates': coordinates,
                                    'orig_elements': [el]
                                })()
                            })()
                            unique_images[element_id] = image_chunk
                            images.append(image_chunk)
                    else:
                        if element_id not in unique_texts:
                            # Get page_number from the original element
                            page_number = None
                            if hasattr(el, 'metadata') and hasattr(el.metadata, 'page_number'):
                                page_number = el.metadata.page_number
                            
                            # Create a new chunk-like object for this text
                            text_chunk = type('TextChunk', (), {
                                'text': element_content,
                                'metadata': type('Metadata', (), {
                                    'text': element_content, 
                                    'page_number': page_number,
                                    'orig_elements': [el]
                                })()
                            })()
                            unique_texts[element_id] = text_chunk
                            texts.append(text_chunk)
            else:
                # Fallback: if no orig_elements, treat the whole chunk as text
                chunk_content = str(chunk)
                chunk_id = hash(chunk_content)
                if chunk_id not in unique_texts:
                    unique_texts[chunk_id] = chunk
                    texts.append(chunk)

        print("="*50)
        print(f"Totale elementi estratti: {len(texts)} testi, {len(images)} immagini, {len(tables)} tabelle")
        print("="*50)

        for text in texts:
                text_elements.append({
                    "text": text,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page": text.metadata.page_number if text.metadata.page_number is not None else 0,
                        "content_type": "text"
                    }
                })
            

        for table in tables:
            try:
                table_counter += 1
                table_id = f"table_{table_counter}"
                page_num = table.metadata.page_number if table.metadata.page_number is not None else 0
                    
                table_element = {
                    "table_html": table.metadata.text_as_html,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page": page_num,
                        "content_type": "table",
                        "table_id": table_id,
                        "table_summary": None  # Placeholder per il riassunto AI
                        }
                }
                # Arricchisci la tabella con il riassunto AI
                table_element = enhance_table_with_summary(table_element)
                table_elements.append(table_element)

            except Exception as e:
                logger.warning(f"Errore nell'elaborazione della tabella: {str(e)}")

        for img_index, img_info in enumerate(images):
                try:
                    page_num = img_info.metadata.page_number if img_info.metadata.page_number is not None else 0
                    width = int(img_info.metadata.coordinates.system.width) if img_info.metadata.coordinates.system.width else 0
                    height = int(img_info.metadata.coordinates.system.height) if img_info.metadata.coordinates.system.height else 0
                    if not is_valid_image(width, height):
                        logger.debug(f"Immagine {img_index+1} pagina {page_num} scartata per dimensioni/qualità")
                        continue
                    
                    image_counter += 1
                    image_id = f"image_{image_counter}"
                    
                    # Log informativo per immagini accettate
                    logger.info(f"Immagine {img_index+1} pagina {page_num} accettata ({image_id})")

                    # Ottinei le informazioni di base sull'immagine
                    image_info_ai = get_comprehensive_image_info(img_info.metadata.image_base64)
                    
                    # Combina caption AI, OCR, e contesto manuale in una descrizione unificata
                    caption_parts = [
                        f"[{image_id}] Descrizione: {image_info_ai['caption']}",
                        f"Testo rilevato: {image_info_ai['ocr_text']}" if image_info_ai["ocr_text"].strip() else None,
                        f"Oggetti rilevati: {', '.join(image_info_ai['detected_objects'])}" if image_info_ai["detected_objects"] else None,
                    ]
                    comprehensive_caption = " | ".join([p for p in caption_parts if p])
                    
                    
                    logger.debug(f"Immagine {img_index+1} pagina {page_num+1} ({image_id}) - Caption: {comprehensive_caption}")
                    
                    image_metadata = {
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "content_type": "image",
                        "image_id": image_id,
                        "image_caption": comprehensive_caption
                    }
                    
                    image_elements.append({
                        "image_base64": img_info.metadata.image_base64,
                        "metadata": image_metadata,
                        "page_content": comprehensive_caption 
                    })
                        
                except Exception as img_e:
                    logger.error(f"Errore processamento immagine {img_index} pagina {page_num}: {str(img_e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Errore durante il parsing del PDF: {str(e)}")
        raise RuntimeError(f"PDF parsing failed: {str(e)}") from e
    
    logger.info(f"Estrazione completata: {len(text_elements)} testi, {len(image_elements)} immagini, {len(table_elements)} tabelle")
    return text_elements, image_elements, table_elements
