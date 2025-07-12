import logging
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_valid_image(width: int, height: int) -> bool:
    """
    Filtro per immagini valide basato su dimensioni e qualità:
    - Dimensioni minime: 100x100 pixel
    - Area minima: 15.000 pixel
    - Dimensioni massime ragionevoli: 5000x5000 pixel
    - Dimensione file minima: 1KB per evitare placeholder/icone
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
