#!/usr/bin/env python3
"""
PDF Parser Core - Versione essenziale per estrazione tabelle, immagini e testo.
Utilizzo: from pdf_parser_core import extract_pdf_content
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from pdf2image import convert_from_path
    from PIL import Image
    import pytesseract
    from groq import Groq
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Dipendenza mancante: {e}")
    print("💡 Installa con: pip install pdf2image pillow pytesseract groq python-dotenv")
    exit(1)

# Carica variabili d'ambiente
load_dotenv()

# Configurazione logging minimale
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class PDFContent:
    """Struttura dati per il contenuto estratto dal PDF."""
    total_pages: int
    total_tables: int
    total_images: int
    total_text_blocks: int
    pages: List[Dict[str, Any]]


class PDFParserCore:
    """Parser PDF essenziale per estrazione tabelle, immagini e testo."""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """
        Inizializza il parser.
        
        Args:
            groq_api_key: Chiave API Groq (opzionale, usa fallback se None)
        """
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.groq_client = None
        
        if self.groq_api_key and self.groq_api_key != 'your_groq_api_key_here':
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception:
                logger.warning("Groq client non disponibile, uso fallback")
    
    def extract_content(self, pdf_path: str, dpi: int = 200) -> PDFContent:
        """
        Estrae contenuto completo dal PDF.
        
        Args:
            pdf_path: Percorso al file PDF
            dpi: Risoluzione per conversione immagini (default: 200)
            
        Returns:
            PDFContent: Oggetto con tutto il contenuto estratto
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File PDF non trovato: {pdf_path}")
        
        # Converti PDF in immagini
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            raise RuntimeError(f"Errore conversione PDF: {e}")
        
        # Analizza ogni pagina
        pages_content = []
        total_tables = 0
        total_images = 0
        total_text_blocks = 0
        
        for page_num, image in enumerate(images, 1):
            page_content = self._analyze_page(image, page_num)
            pages_content.append(page_content)
            
            total_tables += page_content.get('tables', 0)
            total_images += page_content.get('images', 0)
            total_text_blocks += page_content.get('text_blocks', 0)
        
        return PDFContent(
            total_pages=len(images),
            total_tables=total_tables,
            total_images=total_images,
            total_text_blocks=total_text_blocks,
            pages=pages_content
        )
    
    def _analyze_page(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """Analizza una singola pagina."""
        # Estrai testo via OCR
        try:
            text = pytesseract.image_to_string(image, lang='ita+eng')
        except Exception:
            text = ""
        
        # Analisi con LLM se disponibile, altrimenti fallback
        if self.groq_client and text.strip():
            try:
                return self._llm_analysis(text, page_num)
            except Exception:
                pass
        
        return self._fallback_analysis(text, page_num)
    
    def _llm_analysis(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analisi avanzata con LLM."""
        if not self.groq_client:
            return self._fallback_analysis(text, page_num)
            
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
            response = self.groq_client.chat.completions.create(
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
                
                # Valida risultato
                return {
                    'page_number': page_num,
                    'tables': max(0, int(result.get('tables', 0))),
                    'images': max(0, int(result.get('images', 0))),
                    'text_blocks': max(0, int(result.get('text_blocks', 1))),
                    'page_type': result.get('page_type', 'text_content'),
                    'confidence': min(100, max(0, int(result.get('confidence', 70)))),
                    'source': 'llm'
                }
        except Exception:
            pass
        
        return self._fallback_analysis(text, page_num)
    
    def _fallback_analysis(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analisi di fallback senza LLM."""
        if not text.strip():
            return {
                'page_number': page_num,
                'tables': 0,
                'images': 0,
                'text_blocks': 0,
                'page_type': 'empty',
                'confidence': 90,
                'source': 'fallback'
            }
        
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
        
        return {
            'page_number': page_num,
            'tables': tables,
            'images': images,
            'text_blocks': text_blocks,
            'page_type': page_type,
            'confidence': 60,
            'source': 'fallback'
        }


def extract_pdf_content(pdf_path: str, groq_api_key: Optional[str] = None, dpi: int = 200) -> PDFContent:
    """
    Funzione principale per estrazione contenuto PDF.
    
    Args:
        pdf_path: Percorso al file PDF
        groq_api_key: Chiave API Groq (opzionale)
        dpi: Risoluzione conversione (default: 200)
        
    Returns:
        PDFContent: Contenuto estratto (tabelle, immagini, testo)
        
    Example:
        >>> content = extract_pdf_content("document.pdf")
        >>> print(f"Tabelle: {content.total_tables}")
        >>> print(f"Immagini: {content.total_images}")
        >>> for page in content.pages:
        ...     print(f"Pagina {page['page_number']}: {page['tables']} tabelle")
    """
    parser = PDFParserCore(groq_api_key)
    return parser.extract_content(pdf_path, dpi)

