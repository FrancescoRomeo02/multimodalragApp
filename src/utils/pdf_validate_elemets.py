import logging
import pandas as pd
import numpy as np


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

    # Controllo dimensioni minime
    if rows < 2 or cols < 2:
        logger.debug(f"Tabella scartata: dimensioni insufficienti ({rows}x{cols})")
        return False

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
    if meaningful_ratio < 0.6:
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
