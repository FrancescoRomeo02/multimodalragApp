# Esempio in streamlit_app/components/chat.py

from app.pipeline.retriever import create_query_engine
from app.llm.groq_client import get_groq_llm
from app.utils.prompts import qa_template
from llama_index.core import Settings

def get_rag_response(query_str: str):
    """
    Funzione completa che prende una query, esegue il RAG e restituisce la risposta.
    """
    # 1. Configura l'LLM per la generazione
    Settings.llm = get_groq_llm()

    # 2. Crea il motore di query (che si connette a Qdrant)
    query_engine = create_query_engine()

    # 3. Aggiorna il template del prompt nel query engine
    query_engine.update_prompts(
        {"response_synthesis:prompt": qa_template}
    )

    # 4. Esegui la query
    print(f"[*] Esecuzione della query: {query_str}")
    response = query_engine.query(query_str)

    print("[*] Risposta ottenuta.")
    print(f"Risposta: {response}")
    
    # Formattiamo le fonti per una visualizzazione pulita
    sources = []
    for source_node in response.source_nodes:
        sources.append({
            "text": source_node.get_content(),
            "page": source_node.metadata.get("page"),
            "type": source_node.metadata.get("type"),
            "score": source_node.get_score(),
        })

    return {
        "answer": response,
        "sources": sources
    }
