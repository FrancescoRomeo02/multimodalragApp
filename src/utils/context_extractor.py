"""
Modulo per l'estrazione del contesto testuale circostante a tabelle e immagini
"""
import fitz
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextExtractor:
    """Estrae il testo di contesto per tabelle e immagini nei PDF"""
    
    def __init__(self, context_window: int = 500, max_distance: float = 200):
        """
        Args:
            context_window: Numero di caratteri da considerare prima/dopo per il contesto
            max_distance: Distanza massima in punti per considerare un blocco di testo
        """
        self.context_window = context_window
        self.max_distance = max_distance
        
    def get_text_blocks_with_position(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        Estrae blocchi di testo con le loro posizioni sulla pagina
        
        Returns:
            Lista di dizionari con 'text', 'bbox', 'y_center'
        """
        text_blocks = []
        
        # Usa get_text("dict") per avere informazioni dettagliate su posizione
        text_dict = page.get_text("dict")
        
        for block in text_dict["blocks"]:
            if "lines" in block:  # Blocco di testo
                block_text = ""
                block_bbox = block["bbox"]
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                    block_text += " "
                
                if block_text.strip():
                    text_blocks.append({
                        "text": block_text.strip(),
                        "bbox": block_bbox,
                        "y_center": (block_bbox[1] + block_bbox[3]) / 2
                    })
        
        # Ordina i blocchi per posizione verticale
        text_blocks.sort(key=lambda x: x["y_center"])
        return text_blocks
    
    def find_nearest_text_blocks(self, target_bbox: List[float], 
                                text_blocks: List[Dict[str, Any]], 
                                max_distance: Optional[float] = None) -> Tuple[List[str], List[str]]:
        """
        Trova i blocchi di testo più vicini sopra e sotto un elemento
        
        Args:
            target_bbox: Bounding box dell'elemento target [x0, y0, x1, y1]
            text_blocks: Lista dei blocchi di testo con posizioni
            max_distance: Distanza massima in punti per considerare un blocco
            
        Returns:
            Tuple[testi_sopra, testi_sotto]
        """
        if max_distance is None:
            max_distance = self.max_distance
            
        target_y_top = target_bbox[1] 
        target_y_bottom = target_bbox[3]
        target_x_left = target_bbox[0]
        target_x_right = target_bbox[2]
        
        texts_above = []
        texts_below = []
        
        # Prima passa: raccogli tutti i blocchi sopra e sotto
        for block in text_blocks:
            block_y_center = block["y_center"]
            block_bbox = block["bbox"]
            
            # Controlla se c'è sovrapposizione orizzontale (per evitare testi in colonne diverse)
            horizontal_overlap = not (block_bbox[2] < target_x_left or block_bbox[0] > target_x_right)
            
            # Sopra l'elemento
            if block_y_center < target_y_top:
                distance = target_y_top - block_y_center
                if distance <= max_distance and horizontal_overlap:
                    texts_above.append({
                        "text": block["text"],
                        "distance": distance,
                        "y_center": block_y_center
                    })
            
            # Sotto l'elemento
            elif block_y_center > target_y_bottom:
                distance = block_y_center - target_y_bottom
                if distance <= max_distance and horizontal_overlap:
                    texts_below.append({
                        "text": block["text"],
                        "distance": distance,
                        "y_center": block_y_center
                    })
        
        # Ordina per distanza/posizione e prendi i più vicini
        texts_above.sort(key=lambda x: x["y_center"], reverse=True)  # Dal più vicino al più lontano
        texts_below.sort(key=lambda x: x["y_center"])  # Dal più vicino al più lontano
        
        # Prendi fino a 5 blocchi sopra e 5 sotto
        above_texts = [t["text"] for t in texts_above[:5]]
        below_texts = [t["text"] for t in texts_below[:5]]
        
        logger.debug(f"Trovati {len(above_texts)} blocchi sopra e {len(below_texts)} blocchi sotto")
        
        return above_texts, below_texts
    
    def extract_caption(self, texts_above: List[str], texts_below: List[str]) -> Optional[str]:
        """
        Estrae la caption da testi circostanti usando pattern comuni
        
        Args:
            texts_above: Testi sopra l'elemento
            texts_below: Testi sotto l'elemento
            
        Returns:
            Caption estratta o None
        """
        all_texts = texts_above + texts_below
        
        # Pattern di ricerca migliorati
        patterns = [
            # Pattern per figure - più flessibili
            r'(fig(?:ure)?\.?\s*\d+[:\.\-]?\s*[^\n\r.!?]*[.!?])',
            r'(figure\s+\d+[:\.\-]?\s*[^\n\r.!?]*[.!?])',
            
            # Pattern per tabelle - più flessibili  
            r'(tab(?:le|ella)?\.?\s*\d+[:\.\-]?\s*[^\n\r.!?]*[.!?])',
            r'(table\s+\d+[:\.\-]?\s*[^\n\r.!?]*[.!?])',
            
            # Pattern per grafici e diagrammi
            r'(graph\s+\d+[:\.\-]?\s*[^\n\r.!?]*[.!?])',
            r'(chart\s+\d+[:\.\-]?\s*[^\n\r.!?]*[.!?])',
            r'(diagram\s+\d+[:\.\-]?\s*[^\n\r.!?]*[.!?])',
            
            # Pattern generici per elementi numerati
            r'([a-zA-Z]+\s+\d+[:\.\-]\s*[^\n\r.!?]*[.!?])',
        ]
        
        for text in all_texts:
            text_clean = text.strip()
            
            # Prova ogni pattern
            for pattern in patterns:
                match = re.search(pattern, text_clean, re.IGNORECASE | re.MULTILINE)
                if match:
                    caption = match.group(1).strip()
                    # Filtra caption troppo corte o troppo lunghe
                    if 10 <= len(caption) <= 300:
                        logger.debug(f"Caption trovata: {caption}")
                        return caption
            
            # Pattern per frasi descrittive brevi (possibili caption)
            if len(text_clean) < 200 and len(text_clean) > 15:
                # Frasi che iniziano con maiuscola e finiscono con punteggiatura
                if re.match(r'^[A-Z][^.!?]*[.!?]$', text_clean.strip()):
                    # Esclude frasi che sembrano essere corpo del testo
                    if not re.search(r'\b(the|this|these|that|those|in|on|at|by|for|with|as)\b', text_clean.lower()):
                        logger.debug(f"Possibile caption generica: {text_clean}")
                        return text_clean.strip()
        
        return None
    
    def get_context_text(self, texts_above: List[str], texts_below: List[str]) -> str:
        """
        Combina i testi di contesto in una stringa unica
        
        Args:
            texts_above: Testi sopra l'elemento  
            texts_below: Testi sotto l'elemento
            
        Returns:
            Testo di contesto combinato
        """
        context_parts = []
        
        if texts_above:
            # Prendi gli ultimi 3 blocchi sopra (più vicini)
            relevant_above = texts_above[-3:] if len(texts_above) >= 3 else texts_above
            context_parts.append("Contesto precedente: " + " | ".join(relevant_above))
        
        if texts_below:
            # Prendi i primi 3 blocchi sotto (più vicini)
            relevant_below = texts_below[:3] if len(texts_below) >= 3 else texts_below
            context_parts.append("Contesto successivo: " + " | ".join(relevant_below))
        
        return " || ".join(context_parts)
    
    def extract_table_context(self, table_bbox: List[float], page: fitz.Page) -> Dict[str, Optional[str]]:
        """
        Estrae contesto e caption per una tabella usando approccio migliorato
        
        Args:
            table_bbox: Bounding box della tabella
            page: Pagina PDF
            
        Returns:
            Dict con 'caption' e 'context_text'
        """
        # Prova prima il metodo alternativo, poi fallback al metodo standard
        try:
            text_blocks = self.get_full_page_text_with_positions(page)
        except:
            text_blocks = self.get_text_blocks_with_position(page)
        
        texts_above, texts_below = self.find_nearest_text_blocks(table_bbox, text_blocks)
        
        # Log per debug
        logger.debug(f"Tabella bbox: {table_bbox}")
        logger.debug(f"Testi sopra: {len(texts_above)}")
        logger.debug(f"Testi sotto: {len(texts_below)}")
        
        caption = self.extract_caption(texts_above, texts_below)
        context_text = self.get_context_text(texts_above, texts_below) if (texts_above or texts_below) else None
        
        return {
            "caption": caption,
            "context_text": context_text
        }
    
    def extract_image_context(self, image_rect: fitz.Rect, page: fitz.Page) -> Dict[str, Optional[str]]:
        """
        Estrae contesto e caption per un'immagine usando approccio migliorato
        
        Args:
            image_rect: Rettangolo dell'immagine
            page: Pagina PDF
            
        Returns:
            Dict con 'caption' e 'context_text'
        """
        image_bbox = [image_rect.x0, image_rect.y0, image_rect.x1, image_rect.y1]
        
        # Prova prima il metodo alternativo, poi fallback al metodo standard
        try:
            text_blocks = self.get_full_page_text_with_positions(page)
        except:
            text_blocks = self.get_text_blocks_with_position(page)
        
        texts_above, texts_below = self.find_nearest_text_blocks(image_bbox, text_blocks)
        
        # Log per debug
        logger.debug(f"Immagine bbox: {image_bbox}")
        logger.debug(f"Testi sopra: {len(texts_above)}")
        logger.debug(f"Testi sotto: {len(texts_below)}")
        
        caption = self.extract_caption(texts_above, texts_below)
        context_text = self.get_context_text(texts_above, texts_below) if (texts_above or texts_below) else None
        
        return {
            "caption": caption,
            "context_text": context_text
        }
    
    def enhance_text_with_context(self, element_text: str, context_info: Dict[str, Optional[str]]) -> str:
        """
        Combina il testo dell'elemento con il suo contesto per migliorare la ricerca
        
        Args:
            element_text: Testo originale dell'elemento
            context_info: Informazioni di contesto estratte
            
        Returns:
            Testo arricchito con contesto
        """
        enhanced_parts = [element_text]
        
        caption = context_info.get("caption")
        if caption:
            enhanced_parts.append(f"Caption: {caption}")
        
        context_text = context_info.get("context_text")
        if context_text:
            enhanced_parts.append(context_text)
        
        return " | ".join(enhanced_parts)
    
    def get_full_page_text_with_positions(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        Strategia alternativa: estrae tutto il testo della pagina con posizioni dettagliate
        usando get_text("words") per maggiore granularità
        """
        try:
            words = page.get_text("words")  # Lista di (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            
            # Raggruppa le parole in blocchi testuali
            text_blocks = []
            current_block = []
            current_bbox = None
            
            for word_info in words:
                if len(word_info) >= 5:
                    x0, y0, x1, y1, word = word_info[:5]
                    
                    # Inizia un nuovo blocco se è la prima parola o se c'è un gap significativo
                    if not current_block or (current_bbox and y0 > current_bbox[3] + 5):
                        if current_block:
                            # Salva il blocco precedente
                            text = " ".join(current_block)
                            if len(text.strip()) > 3:  # Ignora blocchi troppo corti
                                text_blocks.append({
                                    "text": text.strip(),
                                    "bbox": current_bbox,
                                    "y_center": (current_bbox[1] + current_bbox[3]) / 2
                                })
                        
                        # Inizia nuovo blocco
                        current_block = [word]
                        current_bbox = [x0, y0, x1, y1]
                    else:
                        # Aggiungi al blocco corrente
                        current_block.append(word)
                        # Espandi la bbox
                        current_bbox[0] = min(current_bbox[0], x0)
                        current_bbox[1] = min(current_bbox[1], y0)
                        current_bbox[2] = max(current_bbox[2], x1)
                        current_bbox[3] = max(current_bbox[3], y1)
            
            # Non dimenticare l'ultimo blocco
            if current_block and current_bbox:
                text = " ".join(current_block)
                if len(text.strip()) > 3:
                    text_blocks.append({
                        "text": text.strip(),
                        "bbox": current_bbox,
                        "y_center": (current_bbox[1] + current_bbox[3]) / 2
                    })
            
            logger.debug(f"Estratti {len(text_blocks)} blocchi di testo dalla pagina")
            return text_blocks
            
        except Exception as e:
            logger.warning(f"Errore nell'estrazione alternativa del testo: {e}")
            return self.get_text_blocks_with_position(page)  # Fallback al metodo originale
