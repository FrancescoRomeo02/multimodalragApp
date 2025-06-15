# app/utils/prompts.py
from llama_index.core import PromptTemplate

# Un template di alta qualità per il RAG
# Istruisce l'LLM a basarsi solo sulle fonti, a essere conciso e a citare.
qa_prompt_template_str = (
    "Ti vengono fornite delle informazioni di contesto e una domanda. La tua missione è rispondere alla domanda "
    "utilizzando ESCLUSIVAMENTE le informazioni fornite nel contesto.\n"
    "Non usare alcuna conoscenza pregressa. Se le informazioni non sono nel contesto, rispondi 'Non ho abbastanza informazioni per rispondere'.\n"
    "Sii conciso e diretto. Quando rispondi, cita le fonti indicando il numero di pagina tra parentesi, ad esempio (Pagina 5).\n\n"
    "---------------------\n"
    "Contesto: {context_str}\n"
    "---------------------\n"
    "Domanda: {query_str}\n"
    "Risposta: "
)

# Crea l'oggetto PromptTemplate
qa_template = PromptTemplate(qa_prompt_template_str)