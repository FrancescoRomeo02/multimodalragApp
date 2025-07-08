# src/llm/groq_client.py
from langchain_groq import ChatGroq
from src.config import (
    GROQ_API_KEY, LLM_MODEL_NAME,
    TABLE_SUMMARY_MODEL_SM, TEXT_SUMMARY_MODEL_SM, TEXT_REWRITE_MODEL_SM
)
from pydantic import SecretStr
from typing import Optional

def get_groq_llm(model_name: Optional[str] = None):
    """
    Inizializza e restituisce l'LLM di Groq per LangChain.
    
    Args:
        model_name: Nome del modello specifico da usare. Se None, usa LLM_MODEL_NAME
    """
    if not GROQ_API_KEY:
        raise ValueError("La chiave API di Groq non è stata impostata (GROQ_API_KEY).")
    
    return ChatGroq(
        model=model_name or LLM_MODEL_NAME,
        api_key=SecretStr(GROQ_API_KEY),
        max_tokens=1000,
        stop_sequences=["<|endoftext|>"],
    )

def get_table_summary_llm():
    """Restituisce LLM specifico per riassunto tabelle (versione SM)"""
    return get_groq_llm(TABLE_SUMMARY_MODEL_SM)

def get_text_summary_llm():
    """Restituisce LLM specifico per riassunto testo (versione SM)"""
    return get_groq_llm(TEXT_SUMMARY_MODEL_SM)

def get_text_rewrite_llm():
    """Restituisce LLM specifico per riscrittura testo (versione SM)"""
    return get_groq_llm(TEXT_REWRITE_MODEL_SM)
