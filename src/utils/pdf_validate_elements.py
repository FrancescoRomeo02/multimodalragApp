import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def is_valid_image(width: int, height: int) -> bool:
    """
    Filtro per immagini valide basato su dimensioni e qualità:
    - Dimensioni minime: 120x120 pixel
    - Area minima: 14.400 pixel
    - Dimensioni massime ragionevoli: 5000x5000 pixel
    - Rapporto di aspetto ragionevole (non troppo allungate)
    """
    # Controlli dimensioni di base
    if width < 120 or height < 120:
        logger.debug(f"Immagine scartata: dimensioni troppo piccole ({width}x{height})")
        return False
    
    # Controllo area minima
    area = width * height
    if area < 14400:  # ~120x120 pixel
        logger.debug(f"Immagine scartata: area troppo piccola ({area} pixel)")
        return False
    
    # Controllo dimensioni massime
    if width > 5000 or height > 5000:
        logger.debug(f"Immagine scartata: dimensioni troppo grandi ({width}x{height})")
        return False
    
    # Controllo rapporto di aspetto (evita immagini troppo allungate)
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 10:  # Rapporto massimo 10:1
        logger.debug(f"Immagine scartata: rapporto di aspetto troppo estremo ({aspect_ratio:.2f})")
        return False
    
    return True

def is_valid_table(table_data: Dict[str, Any]) -> bool:
    """
    Filtro per tabelle valide basato su struttura e contenuto:
    - Almeno 2 righe e 2 colonne
    - Contenuto non vuoto
    """
    if not table_data:
        return False
    
    # Se abbiamo metadati sulla struttura della tabella
    if 'rows' in table_data and 'cols' in table_data:
        rows, cols = table_data['rows'], table_data['cols']
        if rows < 2 or cols < 2:
            logger.debug(f"Tabella scartata: dimensioni insufficienti ({rows}x{cols})")
            return False
    
    # Controllo contenuto non vuoto
    content = table_data.get('text', '') or table_data.get('html', '')
    if not content or len(content.strip()) < 20:
        logger.debug("Tabella scartata: contenuto insufficiente")
        return False
    
    return True
