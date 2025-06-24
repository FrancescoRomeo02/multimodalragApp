# app/llm/groq_client.py
from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY, LLM_MODEL_NAME
from pydantic import SecretStr

def get_groq_llm():
    """Inizializza e restituisce l'LLM di Groq per LangChain."""
    if not GROQ_API_KEY:
        raise ValueError("La chiave API di Groq non è stata impostata (GROQ_API_KEY).")
    
    return ChatGroq(
        model=LLM_MODEL_NAME,
        api_key=SecretStr(GROQ_API_KEY),
        max_tokens=1000
    )