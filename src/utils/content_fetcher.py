"""
Questo modulo fornisce funzioni per recuperare contenuti originali da MongoDB
per la visualizzazione nell'interfaccia utente.
"""

import logging
import base64
from typing import Optional, Dict, Any, List
import json

from src.utils.mongodb_utils import mongodb_manager

logger = logging.getLogger(__name__)

def get_original_content_by_mongo_id(mongo_id: str) -> Dict[str, Any]:
    """
    Recupera il contenuto originale da MongoDB dato il suo ID.
    
    Args:
        mongo_id: L'ID MongoDB del documento
        
    Returns:
        Dizionario con il contenuto originale e metadati
    """
    if not mongo_id:
        return {"error": "ID MongoDB non fornito", "content_type": "error"}
    
    document = mongodb_manager.get_document_by_id(mongo_id)
    if not document:
        return {"error": f"Documento con ID {mongo_id} non trovato", "content_type": "error"}
    
    content_type = document.get("content_type", "text")
    
    result = {
        "content_type": content_type,
        "source": document.get("source", "Sconosciuto"),
        "page": document.get("page", "N/A"),
        "mongo_id": mongo_id
    }
    
    # Estrai contenuto specifico per tipo
    if content_type == "image":
        result["caption"] = document.get("caption", "")
        result["image_data"] = document.get("image_data", "")
        
    elif content_type == "table":
        result["table_markdown"] = document.get("table_markdown", "")
        result["table_data"] = document.get("table_data", {})
        
    elif content_type == "text":
        result["content"] = document.get("content", "")
    
    return result

def get_image_by_mongo_id(mongo_id: str) -> Optional[str]:
    """
    Recupera un'immagine originale da MongoDB dato il suo ID.
    
    Args:
        mongo_id: L'ID MongoDB del documento
        
    Returns:
        Stringa base64 dell'immagine o None se non trovata
    """
    document = mongodb_manager.get_document_by_id(mongo_id)
    if not document or document.get("content_type") != "image":
        return None
    
    return document.get("image_data")

def get_table_data_by_mongo_id(mongo_id: str) -> Dict[str, Any]:
    """
    Recupera i dati di una tabella originale da MongoDB dato il suo ID.
    
    Args:
        mongo_id: L'ID MongoDB del documento
        
    Returns:
        Dizionario con i dati della tabella
    """
    document = mongodb_manager.get_document_by_id(mongo_id)
    if not document or document.get("content_type") != "table":
        return {"error": "Tabella non trovata"}
    
    return {
        "table_markdown": document.get("table_markdown", ""),
        "table_data": document.get("table_data", {})
    }

def check_mongo_id_exists(mongo_id: str) -> bool:
    """
    Verifica se un ID MongoDB esiste.
    
    Args:
        mongo_id: L'ID MongoDB da verificare
        
    Returns:
        True se l'ID esiste, False altrimenti
    """
    document = mongodb_manager.get_document_by_id(mongo_id)
    return document is not None
