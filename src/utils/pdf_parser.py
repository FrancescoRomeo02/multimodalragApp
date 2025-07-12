import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from typing import Tuple, List, Dict, Any, Optional
from src.utils.image_info import get_comprehensive_image_info
from src.utils.table_info import enhance_table_with_summary
from unstructured.partition.pdf import partition_pdf
from src.utils.pdf_validate_elements import is_valid_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_metadata_safely(element, *attributes) -> Dict[str, Any]:
    """Estrae metadati in modo sicuro da un elemento."""
    metadata = {}
    if hasattr(element, 'metadata'):
        for attr in attributes:
            if hasattr(element.metadata, attr):
                metadata[attr] = getattr(element.metadata, attr)
    return metadata


def _create_chunk_object(element_content: str, metadata_dict: Dict[str, Any], chunk_type: str):
    """Crea un oggetto chunk dinamico con metadati."""
    return type(f'{chunk_type}Chunk', (), {
        'text': element_content,
        'metadata': type('Metadata', (), metadata_dict)()
    })()


def parse_pdf_elements(pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parser PDF unificato con funzionalità avanzate complete.
    
    FUNZIONALITÀ:
    - Estrazione testo, immagini e tabelle standardizzata per Qdrant
    - Analisi AI delle immagini (caption BLIP, OCR Tesseract, object detection YOLO)
    - Chunking semantico preservato per il testo
    - Deduplicazione intelligente per tutti i tipi di contenuto
    - Metadati completi e strutturati per ogni elemento
    
    Args:
        pdf_path: Percorso al file PDF da processare
        
    Returns:
        Tuple[text_elements, image_elements, table_elements]
    """
    filename = os.path.basename(pdf_path)
    logger.info(f"Avvio parsing PDF: {filename}")
    
    try:
        chunks = partition_pdf(
            filename=pdf_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )

        # Contatori per identificatori univoci
        table_counter = 0
        image_counter = 0
        
        # Collections per elementi unici
        unique_tables = {}
        unique_images = {}
        unique_texts = {}
        
        # Collections finali
        tables, texts, images = [], [], []
        text_elements, image_elements, table_elements = [], [], []

        for chunk in chunks:
            has_special_elements = False
            
            if hasattr(chunk.metadata, 'orig_elements') and chunk.metadata.orig_elements:
                for el in chunk.metadata.orig_elements:
                    element_type = str(type(el))
                    
                    # Determina contenuto per ID unico
                    if hasattr(el, 'text'):
                        element_content = el.text
                    elif hasattr(el, 'to_dict'):
                        element_content = str(el.to_dict())
                    else:
                        element_content = str(el)
                    
                    element_id = hash(element_content)
                    
                    if "Table" in element_type:
                        has_special_elements = True
                        if element_id not in unique_tables:
                            metadata = _extract_metadata_safely(el, 'text_as_html', 'page_number')
                            metadata['orig_elements'] = [el]
                            
                            table_chunk = _create_chunk_object(element_content, metadata, 'Table')
                            unique_tables[element_id] = table_chunk
                            tables.append(table_chunk)
                            
                    elif "Image" in element_type:
                        has_special_elements = True
                        if element_id not in unique_images:
                            metadata = _extract_metadata_safely(el, 'page_number', 'image_base64', 'coordinates')
                            metadata['orig_elements'] = [el]
                            
                            image_chunk = _create_chunk_object(element_content, metadata, 'Image')
                            unique_images[element_id] = image_chunk
                            images.append(image_chunk)
            
            # Preserva chunking semantico per il testo
            if not has_special_elements and hasattr(chunk, 'text') and chunk.text.strip():
                chunk_content = chunk.text
                chunk_id = hash(chunk_content)
                if chunk_id not in unique_texts and len(chunk_content.strip()) > 5:
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
                    image_info_ai = get_comprehensive_image_info(img_info.metadata.image_base64)
                    
                    # Log informativo per immagini accettate
                    logger.info(f"Immagine {img_index+1} pagina {page_num} accettata ({image_id})")

                    # Crea caption completa con AI analysis
                    caption_parts = [
                        f"[{image_id}] Descrizione: {image_info_ai['caption']}",
                        f"Testo rilevato: {image_info_ai['ocr_text']}" if image_info_ai["ocr_text"].strip() else None,
                        f"Oggetti rilevati: {', '.join(image_info_ai['detected_objects'])}" if image_info_ai["detected_objects"] else None,
                    ]
                    comprehensive_caption = " | ".join([p for p in caption_parts if p])
                    
                    logger.debug(f"Immagine {img_index+1} pagina {page_num+1} ({image_id}) - Caption: {comprehensive_caption}")
                    
                    image_metadata = {
                        "source": filename,
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
                    logger.error(f"Errore processamento immagine {img_index}: {str(img_e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Errore durante il parsing del PDF: {str(e)}")
        raise RuntimeError(f"PDF parsing failed: {str(e)}") from e
    
    logger.info(f"Estrazione completata: {len(text_elements)} testi, {len(image_elements)} immagini, {len(table_elements)} tabelle")
    return text_elements, image_elements, table_elements
