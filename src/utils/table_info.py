# file: src/utils/table_info.py

import logging
from typing import Dict, Any, Optional, List
from langchain.schema.messages import HumanMessage
from src.config import GROQ_API_KEY, TABLE_SUMMARY_MODEL_LG
from src.llm.groq_client import get_table_summary_llm

logger = logging.getLogger(__name__)



def create_table_summary(table_html: str, context_info: Optional[Dict[str, str]] = None, table_id: Optional[str] = None) -> str:
    """
    Genera un riassunto intelligente di una tabella usando Groq LLM.
    
    Args:
        table_html: Rappresentazione html della tabella
        context_info: Informazioni di contesto (caption, context_text)
        
    Returns:
        Riassunto testuale della tabella
    """
    try:
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY non configurata, riassunto tabella non disponibile")
            return "Riassunto non disponibile a causa di configurazione mancante."

        # Costruisci il prompt per il riassunto
        table_identifier = f"[{table_id}] " if table_id else ""
        prompt_parts = [
            f"Analizza la seguente tabella {table_identifier} e fornisci un riassunto conciso e informativo che includa:",
            "- Il tipo di dati contenuti",
            "- Le principali tendenze o pattern", 
            "- I valori chiave o interessanti",
            "- Il contesto o scopo della tabella (se evidente)",
            "",
            f"TABELLA {table_identifier}, ecco la tabella in formato html: {table_html}"
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
            f"Fornisci un riassunto chiaro e strutturato in italiano per {table_identifier}:",
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # Usa il modello specifico per riassunto tabelle
        llm = get_table_summary_llm()
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            summary = str(response.content).strip() if response.content else ""
            
            if summary and len(summary) > 10:
                logger.info(f"Riassunto tabella generato con successo usando {TABLE_SUMMARY_MODEL_LG}")
                return summary
            else:
                logger.warning("Riassunto tabella vuoto o troppo breve")
            
            return "Riassunto non disponibile a causa di contenuto insufficiente."
        
        except Exception as llm_error:
            logger.error(f"Errore durante la generazione riassunto tabella con LLM: {llm_error}")
            
            return "Riassunto non disponibile a causa di un errore LLM."
    
    except Exception as e:
        logger.error(f"Errore nel riassunto tabella: {e}")
        
        return "Riassunto non disponibile a causa di un errore interno."
    

def enhance_table_with_summary(table_element: Dict[str, Any]) -> Dict[str, Any]:
    """
    Arricchisce un elemento tabella con il riassunto AI.
    
    Args:
        table_element: Elemento tabella con table_html e metadata

    Returns:
        Elemento tabella arricchito con summary nei metadati
    """
    try:
        table_html = table_element.get('table_html', '')
        metadata = table_element.get('metadata', {})
        
        # Estrai identificatore della tabella
        table_id = metadata.get('table_id')
        
        # Estrai informazioni di contesto dai metadati esistenti
        context_info = {
            'summary': metadata.get('table_summary')
        }
        
        # Genera il riassunto con l'identificatore
        summary = create_table_summary(table_html, context_info, table_id)
        
        # Aggiungi il riassunto ai metadati
        metadata['table_summary'] = summary
        table_element['metadata'] = metadata
        
        logger.info(f"Tabella arricchita con riassunto: {summary[:100]}...")
        return table_element
        
    except Exception as e:
        logger.error(f"Errore nell'arricchimento tabella con riassunto: {e}")
        # Ritorna elemento originale in caso di errore
        return table_element