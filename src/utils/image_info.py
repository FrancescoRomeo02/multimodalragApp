# file: app/utils/image_info.py

import base64
from io import BytesIO
from PIL import Image
from groq import Groq
import logging
import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
from src.config import GROQ_API_KEY, IMG_DESC_MODEL_SM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_detected_objects(base64_str: str) -> list:
    """
    Rileva oggetti in un'immagine usando YOLO.
    
    Args:
        base64_str: Stringa base64 dell'immagine
        
    Returns:
        Una lista di oggetti rilevati con il loro nome e confidenza.
    """
    try:
        model = YOLO("yolov8n.pt") # Carica un modello pre-addestrato
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        results = model(image)
        
        detected_objects = []
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                confidence = float(box.conf)
                detected_objects.append(f"{label}" if confidence > 0.5 else None)

        return [obj for obj in detected_objects if obj is not None]
    except Exception as e:
        logger.error(f"Errore nel rilevamento oggetti: {e}")
        return []


def get_caption(base64_str: str) -> str:
    """
    Genera una descrizione testuale di un'immagine usando Groq's vision model.
    
    Args:
        base64_str: Stringa base64 dell'immagine
        
    Returns:
        Descrizione testuale dell'immagine
    """
    try:
        if not GROQ_API_KEY:
            raise ValueError("La chiave API di Groq non è stata impostata (GROQ_API_KEY).")

        client = Groq(api_key=GROQ_API_KEY)

        # Prova prima con il modello configurato per immagini
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Descrivi dettagliatamente cosa vedi in questa immagine. Fornisci una descrizione accurata e completa degli oggetti, delle persone, del contesto e di qualsiasi testo visibile."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_str}",
                                },
                            },
                        ],
                    }
                ],
                model=IMG_DESC_MODEL_SM,  # Usa il modello configurato per la descrizione immagini
                max_tokens=300,
                temperature=0.1,
            )
            
            caption = chat_completion.choices[0].message.content
            logger.info(f"Caption generata con successo usando {IMG_DESC_MODEL_SM}")
            return caption if caption else "Immagine non descrivibile"
            
        except Exception as vision_error:
            logger.warning(f"Modello {IMG_DESC_MODEL_SM} non supporta immagini: {vision_error}")
            
            # Fallback: prova con modello vision specifico se disponibile
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": "Descrivi dettagliatamente cosa vedi in questa immagine. Fornisci una descrizione accurata e completa degli oggetti, delle persone, del contesto e di qualsiasi testo visibile."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_str}",
                                    },
                                },
                            ],
                        }
                    ],
                    model="llama-3.2-11b-vision-preview",  # Modello vision specifico
                    max_tokens=300,
                    temperature=0.1,
                )
                
                caption = chat_completion.choices[0].message.content
                logger.info(f"Caption generata con fallback usando llama-3.2-11b-vision-preview")
                return caption if caption else "Immagine non descrivibile"
                
            except Exception as fallback_error:
                logger.warning(f"Anche il modello vision fallback è fallito: {fallback_error}")
                raise fallback_error

    except Exception as e:
        logger.error(f"Errore generazione caption con modelli vision: {e}")
        # Fallback finale: usa metodo alternativo con OCR + object detection
        logger.info("Tentativo fallback con metodo alternativo...")
        return get_caption_alternative(base64_str)

def get_image_text(base64_str: str) -> str:
    """
    Estrae testo da un'immagine usando OCR (Tesseract).
    
    Args:
        base64_str: Stringa base64 dell'immagine
        
    Returns:
        Testo estratto dall'immagine
    """
    try:
        # Decodifica e conversione immagine
        image_data = base64.b64decode(base64_str)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Impossibile decodificare immagine")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_string(image_rgb)
        
    except Exception as e:
        logger.error(f"Errore OCR: {e}")
        return ""

def get_comprehensive_image_info(base64_str: str) -> dict:
    """
    Estrae un set completo di informazioni da un'immagine.
    
    Args:
        base64_str: Stringa base64 dell'immagine
        
    Returns:
        Un dizionario contenente tutte le informazioni estratte.
    """
    info = {
        "caption": get_caption(base64_str),
        "ocr_text": get_image_text(base64_str),
        "detected_objects": get_detected_objects(base64_str),
    }
    return info

def get_caption_alternative(base64_str: str) -> str:
    """
    Versione alternativa che combina OCR + analisi oggetti per generare una caption.
    
    Args:
        base64_str: Stringa base64 dell'immagine
        
    Returns:
        Descrizione testuale dell'immagine
    """
    try:
        # Combina informazioni da OCR e object detection
        ocr_text = get_image_text(base64_str)
        detected_objects = get_detected_objects(base64_str)
        
        caption_parts = []
        
        # Aggiungi oggetti rilevati
        if detected_objects:
            unique_objects = list(set(detected_objects))
            if len(unique_objects) == 1:
                caption_parts.append(f"Immagine contenente: {unique_objects[0]}")
            else:
                caption_parts.append(f"Immagine contenente: {', '.join(unique_objects[:3])}")
        
        # Aggiungi testo OCR se presente
        if ocr_text.strip():
            clean_text = ocr_text.strip().replace('\n', ' ')[:100]
            caption_parts.append(f"Con testo visibile: {clean_text}")
        
        # Combina le parti
        if caption_parts:
            return ". ".join(caption_parts)
        else:
            return "Immagine senza contenuto testuale riconoscibile"
            
    except Exception as e:
        logger.error(f"Errore nella caption alternativa: {e}")
        return "Immagine non analizzabile"