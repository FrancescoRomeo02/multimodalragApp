
# app/llm/groq_client.py
import os
from llama_index.llms.groq import Groq

def get_groq_llm(model="llama3-8b-8192"):
    """
    Inizializza e restituisce un'istanza del modello LLM da Groq.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Devi impostare la chiave API: GROQ_API_KEY nel tuo file .env")
    
    return Groq(model=model, api_key=api_key)