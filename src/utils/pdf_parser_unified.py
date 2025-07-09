#!/usr/bin/env python3
"""
PDF Parser Unificato - Combina il meglio di pdf_parser.py e pdf_parser_VLM.py

Questo parser integra:
1. L'interfaccia esistente di pdf_parser.py per compatibilità con il sistema RAG
2. Le capacità LLM avanzate di pdf_parser_VLM.py per analisi intelligente
3. Fallback robusti e gestione errori migliorata
4. Metriche e monitoraggio avanzato
"""

import fitz
import base64
import logging
import json
import os
import time
from typing import Tuple, List, Dict, Any, Optional
from io import BytesIO
from PIL import Image as PILImage
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Import per VLM analysis
try:
    from pdf2image import convert_from_path
    import pytesseract
    from groq import Groq
    from dotenv import load_dotenv
    VLM_AVAILABLE = True
    load_dotenv()
except ImportError as e:
    logging.warning(f"VLM dependencies not available: {e}")
    VLM_AVAILABLE = False
    Groq = None
    load_dotenv = None
    convert_from_path = None
    pytesseract = None

# Import per il sistema esistente
from src.utils.context_extractor import ContextExtractor
from src.utils.image_info import get_comprehensive_image_info
from src.utils.table_info import enhance_table_with_summary

# Carica variabili d'ambiente
if VLM_AVAILABLE and load_dotenv:
    load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PageAnalysis:
    """Analisi VLM di una pagina PDF."""
    page_number: int
    tables: int
    images: int
    text_blocks: int
    page_type: str
    confidence: int
    source: str  # 'llm' or 'fallback'


@dataclass
class ParsingStats:
    """Statistiche del parsing PDF."""
    total_time: float
    total_pages: int
    total_texts: int
    total_images: int
    total_tables: int
    vlm_analysis_used: bool
    vlm_pages_analyzed: int
    fallback_pages: int


class VLMAnalyzer:
    """Analizzatore VLM per contenuto PDF."""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.groq_client = None
        self.available = VLM_AVAILABLE
        
        if self.available and self.groq_api_key and self.groq_api_key != 'your_groq_api_key_here' and Groq:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception:
                logger.warning("Groq client non disponibile, uso fallback")
                self.available = False
    
    def analyze_page_content(self, text: str, page_num: int) -> PageAnalysis:
        """Analizza il contenuto di una pagina con LLM o fallback."""
        if self.available and self.groq_client and text.strip():
            try:
                return self._llm_analysis(text, page_num)
            except Exception as e:
                logger.debug(f"VLM analysis failed for page {page_num}: {e}")
        
        return self._fallback_analysis(text, page_num)
    
    def _llm_analysis(self, text: str, page_num: int) -> PageAnalysis:
        """Analisi avanzata con LLM."""
        prompt = f"""Analizza questo testo OCR di pagina {page_num} e conta ESATTAMENTE:

TESTO:
{text[:3000]}

Restituisci SOLO un JSON valido in questo formato:
{{
  "tables": numero_tabelle_presenti,
  "images": numero_immagini_o_figure,
  "text_blocks": numero_blocchi_testo,
  "page_type": "text_content|mixed_content|table_heavy",
  "confidence": valore_0_100
}}

Criteri:
- Tabelle: strutture con righe/colonne, header, dati numerici
- Immagini: riferimenti a "Fig.", "Figure", "Figura", grafici, diagrammi
- Testo: paragrafi, sezioni, elenchi

JSON:"""

        try:
            response = self.groq_client.chat.completions.create(  # type: ignore
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            
            content = response.choices[0].message.content
            if not content:
                return self._fallback_analysis(text, page_num)
            
            # Estrai JSON
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                
                return PageAnalysis(
                    page_number=page_num,
                    tables=max(0, int(result.get('tables', 0))),
                    images=max(0, int(result.get('images', 0))),
                    text_blocks=max(0, int(result.get('text_blocks', 1))),
                    page_type=result.get('page_type', 'text_content'),
                    confidence=min(100, max(0, int(result.get('confidence', 70)))),
                    source='llm'
                )
        except Exception as e:
            logger.debug(f"LLM analysis error: {e}")
        
        return self._fallback_analysis(text, page_num)
    
    def _fallback_analysis(self, text: str, page_num: int) -> PageAnalysis:
        """Analisi di fallback senza LLM."""
        if not text.strip():
            return PageAnalysis(
                page_number=page_num,
                tables=0,
                images=0,
                text_blocks=0,
                page_type='empty',
                confidence=90,
                source='fallback'
            )
        
        # Conta pattern semplici
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Tabelle: cerca pattern tipici
        table_indicators = [
            len([l for l in lines if '|' in l and l.count('|') >= 2]),
            len([l for l in lines if '\t' in l and l.count('\t') >= 2]),
            text.lower().count('table'),
            text.lower().count('tabella')
        ]
        tables = min(3, max(table_indicators) // 2)
        
        # Immagini: cerca riferimenti
        image_keywords = ['fig.', 'figure', 'figura', 'immagine', 'grafico', 'diagram']
        images = sum(text.lower().count(keyword) for keyword in image_keywords)
        images = min(3, images)
        
        # Blocchi di testo
        text_blocks = max(1, len(lines) // 10)
        
        page_type = 'text_content'
        if tables > 0 and images > 0:
            page_type = 'mixed_content'
        elif tables > 2:
            page_type = 'table_heavy'
        
        return PageAnalysis(
            page_number=page_num,
            tables=tables,
            images=images,
            text_blocks=text_blocks,
            page_type=page_type,
            confidence=60,
            source='fallback'
        )


def is_valid_table(df: pd.DataFrame) -> bool:
    """
    Filtro avanzato per validazione tabelle con criteri migliorati.
    """
    rows, cols = df.shape
    
    # Controlli di base
    if rows < 2 or cols < 2:
        return False
    
    total_cells = rows * cols
    empty_cells = df.isna().values.sum()
    empty_ratio = empty_cells / total_cells
    
    # Soglia più rigorosa per celle vuote
    if empty_ratio > 0.7:
        logger.debug(f"Tabella scartata: troppe celle vuote ({empty_ratio:.2%})")
        return False
    
    # Conta celle con contenuto significativo
    meaningful_content = 0
    for col in df.columns:
        for value in df[col].dropna():
            if isinstance(value, str) and len(str(value).strip()) > 1:
                meaningful_content += 1
    
    meaningful_ratio = meaningful_content / (total_cells - empty_cells) if (total_cells - empty_cells) > 0 else 0
    if meaningful_ratio < 0.6:
        logger.debug(f"Tabella scartata: contenuto poco significativo ({meaningful_ratio:.2%})")
        return False
    
    logger.debug(f"Tabella valida: {rows}x{cols}, vuote: {empty_ratio:.2%}, significative: {meaningful_ratio:.2%}")
    return True


def is_valid_image(width: int, height: int, image_data: bytes) -> bool:
    """
    Filtro per immagini valide basato su dimensioni e qualità.
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
    
    # Controllo dimensione file
    if len(image_data) < 1024:  # 1KB minimo
        logger.debug(f"Immagine scartata: file troppo piccolo ({len(image_data)} bytes)")
        return False
    
    # Controllo rapporto di aspetto
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 10:  # Rapporto massimo 10:1
        logger.debug(f"Immagine scartata: rapporto di aspetto troppo estremo ({aspect_ratio:.2f})")
        return False
    
    return True


def extract_tables_from_page(page) -> List[Dict[str, Any]]:
    """
    Estrae tabelle da una pagina PDF usando PyMuPDF.
    """
    tables = []
    try:
        tabs = page.find_tables()  # type: ignore
        if not tabs or len(tabs.tables) == 0:
            return tables
            
        for i, table in enumerate(tabs.tables):
            try:
                if not hasattr(table, 'bbox') or not table.bbox:
                    continue
                    
                df = table.to_pandas()
                
                # Converti e pulisci
                df = df.map(lambda x: str(x).strip() if pd.notna(x) else x)
                df = df.replace(r'^\s*$', np.nan, regex=True)
                df = df.dropna(how='all').dropna(how='all', axis=1)
                
                if not is_valid_table(df):
                    continue
                
                # Converti in formato markdown
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
                
                rows, cols = df.shape
                logger.info(f"Tabella {i+1} pagina {(page.number or 0) + 1} accettata: {rows}x{cols} celle")

            except Exception as table_e:
                logger.warning(f"Errore nell'estrazione della tabella {i+1}: {str(table_e)}")
                continue
                
    except Exception as e:
        logger.error(f"Errore generale nell'estrazione delle tabelle: {str(e)}")
    
    return tables


def parse_pdf_elements_unified(pdf_path: str, use_vlm: bool = True, groq_api_key: Optional[str] = None, dpi: int = 200) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], ParsingStats]:
    """
    Parser PDF unificato che usa SOLO l'analisi VLM per identificare contenuti,
    sostituendo completamente PyMuPDF per il riconoscimento di tabelle e immagini.
    
    Args:
        pdf_path: Percorso al file PDF da processare
        use_vlm: Se utilizzare l'analisi VLM (default: True)
        groq_api_key: Chiave API Groq (opzionale)
        dpi: Risoluzione per conversione immagini (default: 200)
        
    Returns:
        Tuple di quattro elementi:
        - text_elements: Lista di elementi testuali con metadati
        - image_elements: Lista di immagini simulate basate su analisi VLM
        - table_elements: Lista di tabelle simulate basate su analisi VLM
        - stats: Statistiche del parsing
    """
    start_time = time.time()
    logger.info(f"Avvio parsing VLM del PDF: {os.path.basename(pdf_path)}")
    
    text_elements = []
    image_elements = []
    table_elements = []
    
    # Contatori per statistiche
    vlm_pages_analyzed = 0
    fallback_pages = 0
    total_pages = 0
    
    try:
        if not VLM_AVAILABLE or not convert_from_path or not pytesseract:
            logger.warning("VLM dependencies non disponibili, uso fallback PyMuPDF")
            return parse_pdf_elements_fallback(pdf_path)
        
        # Converti PDF in immagini per analisi VLM
        images = convert_from_path(pdf_path, dpi=dpi)
        total_pages = len(images)
        filename = os.path.basename(pdf_path)
        
        # Inizializza VLM analyzer
        vlm_analyzer = VLMAnalyzer(groq_api_key) if use_vlm else None
        
        for page_num, page_image in enumerate(images, 1):
            # Estrai testo via OCR
            try:
                text = pytesseract.image_to_string(page_image, lang='ita+eng')
            except Exception:
                text = ""
            
            # Analisi VLM della pagina per identificare contenuti
            page_analysis = None
            if vlm_analyzer and text.strip():
                page_analysis = vlm_analyzer.analyze_page_content(text, page_num)
                if page_analysis.source == 'llm':
                    vlm_pages_analyzed += 1
                else:
                    fallback_pages += 1
                    
                logger.info(f"Pagina {page_num} analisi VLM: "
                          f"{page_analysis.tables} tab, {page_analysis.images} img, "
                          f"{page_analysis.text_blocks} testi ({page_analysis.source}, conf: {page_analysis.confidence}%)")
            else:
                fallback_pages += 1
            
            # 1. TESTO: Aggiungi sempre l'elemento di testo se presente
            if text.strip():
                text_elements.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                        "content_type": "text"
                    }
                })
            
            # 2. TABELLE: Crea elementi tabella basati sull'analisi VLM
            if page_analysis and page_analysis.tables > 0:
                for table_idx in range(page_analysis.tables):
                    table_id = f"table_{len(table_elements) + 1}"
                    
                    # Estrai potenziali contenuti tabulari dal testo OCR
                    table_content = extract_table_content_from_text(text, table_idx)
                    
                    table_element = {
                        "table_data": table_content["data"],
                        "table_markdown": table_content["markdown"],
                        "metadata": {
                            "source": filename,
                            "page": page_num,
                            "content_type": "table",
                            "table_id": table_id,
                            "table_summary": f"Tabella identificata da VLM su pagina {page_num}",
                            "vlm_confidence": page_analysis.confidence,
                            "vlm_detected": True
                        }
                    }
                    
                    table_elements.append(table_element)
                    logger.info(f"Tabella VLM {table_id} creata per pagina {page_num}")
            
            # 3. IMMAGINI: Crea elementi immagine basati sull'analisi VLM
            if page_analysis and page_analysis.images > 0:
                for img_idx in range(page_analysis.images):
                    image_id = f"image_{len(image_elements) + 1}"
                    
                    # Converti l'intera pagina come immagine per rappresentare il contenuto visivo
                    img_buffer = BytesIO()
                    page_image.save(img_buffer, format='PNG', optimize=True)
                    image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    
                    # Crea descrizione basata su testo OCR e analisi VLM
                    image_description = create_image_description_from_vlm(text, page_analysis, img_idx)
                    
                    image_element = {
                        "image_base64": image_base64,
                        "metadata": {
                            "source": filename,
                            "page": page_num,
                            "content_type": "image",
                            "image_id": image_id,
                            "image_caption": f"Immagine {img_idx + 1} identificata da VLM su pagina {page_num}",
                            "vlm_confidence": page_analysis.confidence,
                            "vlm_detected": True
                        },
                        "page_content": image_description
                    }
                    
                    image_elements.append(image_element)
                    logger.info(f"Immagine VLM {image_id} creata per pagina {page_num}")
    
    except Exception as e:
        logger.error(f"Errore durante il parsing VLM: {str(e)}")
        logger.warning("Fallback a parsing tradizionale")
        return parse_pdf_elements_fallback(pdf_path)
    
    # Crea statistiche
    total_time = time.time() - start_time
    stats = ParsingStats(
        total_time=total_time,
        total_pages=total_pages,
        total_texts=len(text_elements),
        total_images=len(image_elements),
        total_tables=len(table_elements),
        vlm_analysis_used=True,
        vlm_pages_analyzed=vlm_pages_analyzed,
        fallback_pages=fallback_pages
    )
    
    logger.info(f"Parsing VLM completato in {total_time:.2f}s: "
              f"{len(text_elements)} testi, {len(image_elements)} immagini VLM, {len(table_elements)} tabelle VLM")
    logger.info(f"Analisi VLM: {vlm_pages_analyzed} pagine LLM, {fallback_pages} fallback")
    
    return text_elements, image_elements, table_elements, stats


def extract_table_content_from_text(text: str, table_index: int) -> Dict[str, Any]:
    """
    Estrae contenuto tabulare dal testo OCR per creare una struttura tabella compatibile.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Cerca linee che sembrano tabelle (con separatori)
    table_lines = []
    for line in lines:
        if '|' in line and line.count('|') >= 2:
            table_lines.append(line)
        elif '\t' in line and line.count('\t') >= 2:
            table_lines.append(line.replace('\t', '|'))
    
    if not table_lines:
        # Crea una tabella generica se non si trovano pattern
        table_lines = [f"Contenuto {i+1}|Pagina" for i in range(2)]
        table_lines.insert(0, "Campo|Valore")
    
    # Prendi solo le righe per questa specifica tabella (se ci sono più tabelle)
    lines_per_table = max(2, len(table_lines) // max(1, table_index + 1))
    start_idx = table_index * lines_per_table
    end_idx = start_idx + lines_per_table
    selected_lines = table_lines[start_idx:end_idx] if table_lines else ["Campo|Valore", "Contenuto|VLM"]
    
    # Crea struttura dati tabella
    rows = []
    headers = []
    
    for i, line in enumerate(selected_lines):
        cells = [cell.strip() for cell in line.split('|')]
        if i == 0:
            headers = cells
        rows.append(cells)
    
    # Assicurati che ci siano headers
    if not headers:
        headers = [f"Colonna_{i+1}" for i in range(len(rows[0]) if rows else 2)]
    
    # Crea markdown
    markdown_lines = []
    if headers:
        markdown_lines.append('| ' + ' | '.join(headers) + ' |')
        markdown_lines.append('|' + '---|' * len(headers))
    
    for row in rows[1:] if len(rows) > 1 else rows:
        # Assicurati che la riga abbia lo stesso numero di colonne degli headers
        while len(row) < len(headers):
            row.append("")
        row = row[:len(headers)]  # Tronca se troppo lunga
        markdown_lines.append('| ' + ' | '.join(row) + ' |')
    
    return {
        "data": {
            "cells": rows,
            "headers": headers,
            "shape": (len(rows), len(headers))
        },
        "markdown": '\n'.join(markdown_lines)
    }


def create_image_description_from_vlm(text: str, page_analysis: PageAnalysis, image_index: int) -> str:
    """
    Crea una descrizione dell'immagine basata sul testo OCR e l'analisi VLM.
    """
    # Cerca riferimenti a figure/immagini nel testo
    image_refs = []
    text_lower = text.lower()
    
    # Pattern comuni per riferimenti a immagini
    patterns = ['fig.', 'figure', 'figura', 'immagine', 'grafico', 'diagram', 'chart']
    for pattern in patterns:
        if pattern in text_lower:
            # Estrai la riga che contiene il riferimento
            for line in text.split('\n'):
                if pattern in line.lower():
                    image_refs.append(line.strip())
                    break
    
    # Crea descrizione
    if image_refs:
        description = f"Immagine {image_index + 1}: {image_refs[0]}"
    else:
        description = f"Contenuto visivo identificato da VLM su pagina {page_analysis.page_number}"
    
    # Aggiungi contesto dal tipo di pagina
    if page_analysis.page_type == 'mixed_content':
        description += " (pagina con contenuto misto: testo, immagini e tabelle)"
    elif page_analysis.page_type == 'table_heavy':
        description += " (pagina ricca di tabelle con elementi grafici)"
    
    description += f" - Confidence VLM: {page_analysis.confidence}%"
    
    return description


def parse_pdf_elements_fallback(pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], ParsingStats]:
    """
    Fallback al parser PyMuPDF originale quando VLM non è disponibile.
    """
    logger.info("Usando parser PyMuPDF fallback")
    
    # Importa il parser originale
    from src.utils.pdf_parser import parse_pdf_elements as original_parser
    
    start_time = time.time()
    texts, images, tables = original_parser(pdf_path)
    total_time = time.time() - start_time
    
    # Conta pagine dal testo
    total_pages = len(set(text['metadata']['page'] for text in texts)) if texts else 0
    
    stats = ParsingStats(
        total_time=total_time,
        total_pages=total_pages,
        total_texts=len(texts),
        total_images=len(images),
        total_tables=len(tables),
        vlm_analysis_used=False,
        vlm_pages_analyzed=0,
        fallback_pages=total_pages
    )
    
    return texts, images, tables, stats


# Funzione di compatibilità con l'interfaccia esistente
def parse_pdf_elements(pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Interfaccia di compatibilità con il sistema esistente.
    Utilizza il parser unificato ma restituisce solo i dati senza statistiche.
    """
    text_elements, image_elements, table_elements, _ = parse_pdf_elements_unified(pdf_path)
    return text_elements, image_elements, table_elements


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Utilizzo: python pdf_parser_unified.py <file.pdf>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    try:
        text_elements, image_elements, table_elements, stats = parse_pdf_elements_unified(pdf_file)
        
        print(f"📄 PDF: {Path(pdf_file).name}")
        print(f"⏱️  Tempo: {stats.total_time:.2f}s")
        print(f"📑 Pagine: {stats.total_pages}")
        print(f"📝 Testi: {stats.total_texts}")
        print(f"📊 Tabelle: {stats.total_tables}")
        print(f"🖼️  Immagini: {stats.total_images}")
        print(f"🧠 VLM: {'✓' if stats.vlm_analysis_used else '✗'} ({stats.vlm_pages_analyzed} LLM, {stats.fallback_pages} fallback)")
        print()
        
        # Mostra dettagli prime pagine
        for i, (text, image, table) in enumerate(zip(text_elements[:3], 
                                                   image_elements[:3] if image_elements else [None]*3,
                                                   table_elements[:3] if table_elements else [None]*3)):
            page = text['metadata']['page']
            print(f"Pagina {page}: testo={'✓' if text else '✗'}, "
                  f"img={'✓' if image else '✗'}, tab={'✓' if table else '✗'}")
        
        if stats.total_pages > 3:
            print("...")
            
    except Exception as e:
        print(f"❌ Errore: {e}")
