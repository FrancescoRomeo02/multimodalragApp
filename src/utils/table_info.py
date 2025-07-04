# file: src/utils/table_info.py

import logging
from typing import Dict, Any, Optional, List
from groq import Groq
from langchain.schema.messages import HumanMessage
from src.config import GROQ_API_KEY, TABLE_SUMMARY_MODEL_SM
from src.llm.groq_client import get_table_summary_llm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sanitize_table_cells(tables_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converte tutti i valori delle celle delle tabelle in stringa per evitare errori di validazione.
    Garantisce che i dati della tabella siano serializzabili e uniformi.
    
    Args:
        tables_dicts: Lista di dizionari contenenti i dati delle tabelle
        
    Returns:
        Lista di dizionari con i valori delle celle convertiti in stringa
    """
    for table in tables_dicts:
        cells = table.get("table_data", {}).get("cells", [])
        for i, row in enumerate(cells):
            for j, cell in enumerate(row):
                if cell is None:
                    cells[i][j] = ""
                elif isinstance(cell, float):
                    # Converti float in stringa senza notazione scientifica
                    cells[i][j] = f"{cell:.6f}".rstrip('0').rstrip('.') if '.' in f"{cell:.6f}" else str(cell)
                elif not isinstance(cell, str):
                    cells[i][j] = str(cell)
    return tables_dicts


def create_table_summary(table_markdown: str, table_data: Dict[str, Any], context_info: Optional[Dict[str, str]] = None, table_id: Optional[str] = None) -> str:
    """
    Genera un riassunto intelligente di una tabella usando Groq LLM.
    
    Args:
        table_markdown: Rappresentazione markdown della tabella
        table_data: Dati strutturati della tabella (cells, headers, shape)
        context_info: Informazioni di contesto (caption, context_text)
        
    Returns:
        Riassunto testuale della tabella
    """
    try:
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY non configurata, uso fallback per riassunto tabella")
            return create_table_summary_fallback(table_markdown, table_data, context_info)

        # Costruisci il prompt per il riassunto
        table_identifier = f"[{table_id}] " if table_id else ""
        prompt_parts = [
            f"Analizza la seguente tabella {table_identifier}e fornisci un riassunto conciso e informativo che includa:",
            "- Il tipo di dati contenuti",
            "- Le principali tendenze o pattern", 
            "- I valori chiave o interessanti",
            "- Il contesto o scopo della tabella (se evidente)",
            "",
            f"TABELLA {table_identifier}({table_data.get('shape', 'N/A')} celle):",
            table_markdown
        ]
        
        # Aggiungi informazioni di contesto se disponibili
        if context_info:
            if context_info.get('summary'):
                prompt_parts.extend([
                    "",
                    f"RIASSUNTO ESISTENTE: {context_info['summary']}"
                ])
        
        prompt_parts.extend([
            "",
            f"Fornisci un riassunto chiaro e strutturato in italiano per {table_identifier}(massimo 200 parole):",
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # Usa il modello specifico per riassunto tabelle
        llm = get_table_summary_llm()
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            summary = str(response.content).strip() if response.content else ""
            
            if summary and len(summary) > 10:
                logger.info(f"Riassunto tabella generato con successo usando {TABLE_SUMMARY_MODEL_SM}")
                return summary
            else:
                logger.warning("Riassunto tabella vuoto o troppo breve, uso fallback")
                return create_table_summary_fallback(table_markdown, table_data, context_info, table_id)
        except Exception as llm_error:
            logger.error(f"Errore durante la generazione riassunto tabella con LLM: {llm_error}")
            return create_table_summary_fallback(table_markdown, table_data, context_info, table_id)
    
    except Exception as e:
        logger.error(f"Errore nel riassunto tabella: {e}")
        return create_table_summary_fallback(table_markdown, table_data, context_info)
        


def create_table_summary_fallback(table_markdown: str, table_data: Dict[str, Any], context_info: Optional[Dict[str, str]] = None, table_id: Optional[str] = None) -> str:
    """
    Genera un riassunto basico della tabella senza LLM (fallback).
    
    Args:
        table_markdown: Rappresentazione markdown della tabella
        table_data: Dati strutturati della tabella
        context_info: Informazioni di contesto opzionali
        
    Returns:
        Riassunto basico della tabella
    """
    try:
        shape = table_data.get('shape', (0, 0))
        headers = table_data.get('headers', [])
        cells = table_data.get('cells', [])
        
        summary_parts = []
        
        # Aggiungi identificatore se disponibile
        table_identifier = f"[{table_id}] " if table_id else ""
        
        # Informazioni base
        summary_parts.append(f"{table_identifier}Tabella con {shape[0]} righe e {shape[1]} colonne")
        
        # Headers
        if headers:
            summary_parts.append(f"Colonne: {', '.join(headers[:5])}")  # Max 5 headers
            if len(headers) > 5:
                summary_parts[-1] += f" (e altre {len(headers) - 5})"
        
        # Analisi dati semplice
        if cells and len(cells) > 1:  # Escludi header row
            data_rows = cells[1:] if headers else cells
            non_empty_cells = sum(1 for row in data_rows for cell in row if cell and str(cell).strip())
            total_data_cells = len(data_rows) * len(data_rows[0]) if data_rows else 0
            
            if total_data_cells > 0:
                fill_rate = (non_empty_cells / total_data_cells) * 100
                summary_parts.append(f"Completezza dati: {fill_rate:.0f}%")
        
        # Aggiungi informazioni di base se disponibili
        if context_info and context_info.get('summary'):
            summary_parts.append(f"Riassunto: {context_info['summary']}")
        
        # Prova a identificare il tipo di contenuto
        content_type = identify_table_content_type(headers, cells)
        if content_type:
            summary_parts.append(f"Tipo: {content_type}")
        
        return ". ".join(summary_parts) + "."
        
    except Exception as e:
        logger.error(f"Errore nel fallback riassunto tabella: {e}")
        return f"Tabella con {table_data.get('shape', 'N/A')} elementi"


def identify_table_content_type(headers: List[str], cells: List[List]) -> Optional[str]:
    """
    Prova a identificare il tipo di contenuto della tabella basandosi su headers e dati.
    """
    try:
        if not headers:
            return None
            
        headers_lower = [h.lower() if h else "" for h in headers]
        
        # Pattern comuni
        if any(word in " ".join(headers_lower) for word in ["name", "nome", "cognome", "person", "persona"]):
            return "anagrafica/persone"
        elif any(word in " ".join(headers_lower) for word in ["price", "cost", "prezzo", "costo", "euro", "$"]):
            return "dati finanziari"
        elif any(word in " ".join(headers_lower) for word in ["date", "data", "time", "tempo", "anno", "year"]):
            return "dati temporali"
        elif any(word in " ".join(headers_lower) for word in ["result", "risultato", "score", "punteggio", "performance"]):
            return "risultati/performance"
        elif any(word in " ".join(headers_lower) for word in ["product", "prodotto", "item", "articolo"]):
            return "catalogo prodotti"
        
        # Controlla se prevalentemente numerica
        if cells and len(cells) > 1:
            numeric_count = 0
            total_count = 0
            
            for row in cells[1:]:  # Salta header se presente
                for cell in row:
                    if cell is not None and str(cell).strip():
                        total_count += 1
                        try:
                            float(str(cell).replace(',', '.'))
                            numeric_count += 1
                        except:
                            pass
            
            if total_count > 0 and numeric_count / total_count > 0.7:
                return "dati numerici"
        
        return None
        
    except Exception:
        return None


def enhance_table_with_summary(table_element: Dict[str, Any]) -> Dict[str, Any]:
    """
    Arricchisce un elemento tabella con il riassunto AI.
    
    Args:
        table_element: Elemento tabella con table_markdown, table_data, metadata
        
    Returns:
        Elemento tabella arricchito con summary nei metadati
    """
    try:
        table_markdown = table_element.get('table_markdown', '')
        table_data = table_element.get('table_data', {})
        metadata = table_element.get('metadata', {})
        
        # Estrai identificatore della tabella
        table_id = metadata.get('table_id')
        
        # Estrai informazioni di contesto dai metadati esistenti
        context_info = {
            'summary': metadata.get('table_summary')
        }
        
        # Genera il riassunto con l'identificatore
        summary = create_table_summary(table_markdown, table_data, context_info, table_id)
        
        # Aggiungi il riassunto ai metadati
        metadata['table_summary'] = summary
        table_element['metadata'] = metadata
        
        logger.info(f"Tabella arricchita con riassunto: {summary[:100]}...")
        return table_element
        
    except Exception as e:
        logger.error(f"Errore nell'arricchimento tabella con riassunto: {e}")
        # Ritorna elemento originale in caso di errore
        return table_element