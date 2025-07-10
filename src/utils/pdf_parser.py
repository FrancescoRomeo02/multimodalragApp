import os
import time
import groq
from typing import List, Any, Dict
from dotenv import load_dotenv
import logging
import json

# Configura un logging di base per vedere meglio gli errori
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carica variabili ambiente (GROQ_API_KEY)
load_dotenv()

# === 1. Partizionamento PDF con Unstructured ===
from unstructured.partition.pdf import partition_pdf
from dataclasses import dataclass, field, asdict

@dataclass
class TextChunk:
    text: str
    metadata: dict = field(default_factory=dict)
    content_type: str = "text"

def parse_pdf_elements(pdf_path: str) -> List[TextChunk]:

    # --- CONFIGURAZIONE ---
    file_path = pdf_path
    source_filename = os.path.basename(pdf_path)  # Estrae il nome del file per usarlo come source

    if not os.path.exists(file_path):
        logging.error(f"ERRORE: Il file {file_path} non è stato trovato.")
        exit()

    logging.info("Avvio del partizionamento del PDF...")
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables

        extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="by_title",          # or 'basic'
        max_characters=10000,                  # defaults to 500
        combine_text_under_n_chars=2000,       # defaults to 0
        new_after_n_chars=6000,
    )
    logging.info("Partizionamento completato.")

    # === 2. Separazione elementi e preparazione per riassunti ===
    tables = []
    texts = []
    images = []

    seen_texts = set()
    seen_tables = set()

    for chunk in chunks:
        orig_elements = getattr(chunk.metadata, "orig_elements", None)
        if orig_elements is not None:
            for el in orig_elements:
                el_type = type(el).__name__

                # Prepara i metadati di base
                metadata = {
                    "source": source_filename,
                    "page": getattr(chunk.metadata, "page_number", None)
                }
                
                if el_type == "Table":
                    if chunk.text not in seen_tables:
                        tables.append(TextChunk(
                            text=chunk.text,
                            metadata=metadata,
                            content_type="table"
                        ))
                        seen_tables.add(chunk.text)

                elif el_type == "Image":
                    base64_img = el.metadata.image_base64
                    if base64_img:
                        image_metadata = metadata.copy()
                        image_metadata["image_present"] = True
                        images.append({
                            "image_base64": base64_img,
                            "metadata": image_metadata
                        })

                else:
                    if chunk.text not in seen_texts:
                        texts.append(TextChunk(
                            text=chunk.text,
                            metadata=metadata,
                            content_type="text"
                        ))
                        seen_texts.add(chunk.text)

    logging.info(f"Elementi estratti: {len(texts)} testi, {len(tables)} tabelle, {len(images)} immagini.")

    # === 3. Setup LangChain ===
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_groq import ChatGroq
    from langchain_core.runnables import RunnableLambda, RunnableConfig

    # === 3A. Riassunto testi e tabelle ===
    prompt_text_template = """
    Sei un assistente incaricato di riassumere tabelle e testi in modo conciso.
    Fornisci solo il riassunto, senza commenti aggiuntivi.

    Elemento da riassumere:
    {element}
    """
    prompt_groq = ChatPromptTemplate.from_template(prompt_text_template)
    model_groq = ChatGroq(temperature=0.1, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x.text} | prompt_groq | model_groq | StrOutputParser()

    # === 3B. Riassunto immagini (via Groq multimodale) ===
    client = groq.Client()
    prompt_image_text = """
    Descrivi l'immagine in dettaglio. Concentrati sul suo contenuto, come grafici, diagrammi o elementi visivi chiave.
    Se è un grafico o una tabella, spiega cosa rappresenta.
    Rispondi solo con la descrizione, senza commenti aggiuntivi.
    """

    def summarize_image_with_groq(image_obj: dict) -> str:
        """
        Funzione per riassumere un singolo elemento Immagine usando il client Groq.
        Accetta un oggetto contenente l'immagine base64 e i metadati.
        """
        try:
            # Estraiamo la stringa base64 e i metadati dall'oggetto
            image_b64 = image_obj["image_base64"]
            
            response = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_image_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=400,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Errore durante il riassunto dell'immagine: {str(e)}"
            
    # Funzione che trasforma un riassunto di immagine in un TextChunk
    def image_summary_to_textchunk(image_obj: dict, summary: str) -> TextChunk:
        metadata = image_obj.get("metadata", {})
        return TextChunk(
            text=summary,
            metadata=metadata,
            content_type="image_description"
        )
        
    image_summarize_chain = RunnableLambda(summarize_image_with_groq)

    # === 4. Esecuzione con Batching e Retry Dinamico ===
    def process_in_batch(
        chain: Any, 
        items: List[Any], 
        max_concurrency: int = 5, 
        max_retries: int = 3, 
        initial_delay: float = 5.0
    ) -> List[str]:
        """
        Processa una lista di elementi usando chain.batch() con una logica di retry
        per gestire i RateLimitError in modo dinamico (backoff esponenziale).
        """
        if not items:
            return []

        # Mappa per tenere traccia degli indici originali degli elementi da processare
        items_to_process = list(enumerate(items))
        final_results = [""] * len(items)
        config = RunnableConfig(max_concurrency=max_concurrency)
        
        for attempt in range(max_retries + 1):
            if not items_to_process:
                break # Tutti gli elementi sono stati processati con successo

            logging.info(f"Tentativo {attempt + 1}/{max_retries + 1}. Elementi da processare: {len(items_to_process)}")
            
            # Estrai gli indici e gli oggetti per il batch corrente
            indices, batch_items = zip(*items_to_process)
            
            # Esegui il batch
            results = chain.batch(batch_items, config=config, return_exceptions=True)
            
            failed_items_for_next_round = []
            rate_limit_hit = False

            for i, res in enumerate(results):
                original_index = indices[i]
                if isinstance(res, groq.RateLimitError):
                    # Se è un errore di rate limit, aggiungilo alla lista per il prossimo tentativo
                    failed_items_for_next_round.append(items_to_process[i])
                    rate_limit_hit = True
                elif isinstance(res, Exception):
                    # Se è un altro tipo di errore, consideralo un fallimento permanente
                    logging.error(f"Errore non recuperabile sull'elemento {original_index}: {res}")
                    final_results[original_index] = f"Errore permanente: {res}"
                else:
                    # Successo! Salva il risultato
                    logging.info(f"  ✓ Elaborato elemento con indice originale {original_index}")
                    final_results[original_index] = res
            
            # Prepara per il prossimo tentativo
            items_to_process = failed_items_for_next_round
            
            if items_to_process and rate_limit_hit and attempt < max_retries:
                delay = initial_delay * (2 ** attempt)
                logging.warning(f"Rate limit raggiunto. Attendo {delay:.1f} secondi prima di riprovare con {len(items_to_process)} elementi.")
                time.sleep(delay)

        # Dopo tutti i tentativi, segna come falliti gli elementi rimasti
        if items_to_process:
            for original_index, _ in items_to_process:
                logging.error(f"Fallimento finale per l'elemento {original_index} dopo {max_retries + 1} tentativi.")
                final_results[original_index] = f"Errore: Fallito dopo {max_retries + 1} tentativi di rate limit."

        return final_results



    # === 6. Esecuzione e creazione dei chunk di testo finale ===
    final_text_chunks = []
    
    # --- Processa il testo ---
    if texts:
        logging.info("\n--- Inizio riassunto testi ---")
        text_summaries = process_in_batch(summarize_chain, texts, max_concurrency=5)
        
        # Crea TextChunk per ogni riassunto di testo
        for i, summary in enumerate(text_summaries):
            if summary and not summary.startswith("Errore"):
                # Assicurati che i metadati contengano almeno source e page
                metadata = {
                    "source": texts[i].metadata.get("source", source_filename),
                    "page": texts[i].metadata.get("page", None)
                }
                chunk = TextChunk(
                    text=summary,
                    metadata=metadata,
                    content_type="text"  # Usa sempre "text" per uniformità
                )
                final_text_chunks.append(chunk)
    
    # --- Processa le tabelle ---
    if tables:
        logging.info("\n--- Inizio riassunto tabelle ---")
        table_summaries = process_in_batch(summarize_chain, tables, max_concurrency=5)
        
        # Crea TextChunk per ogni riassunto di tabella
        for i, summary in enumerate(table_summaries):
            if summary and not summary.startswith("Errore"):
                # Assicurati che i metadati contengano almeno source e page
                metadata = {
                    "source": tables[i].metadata.get("source", source_filename),
                    "page": tables[i].metadata.get("page", None)
                }
                chunk = TextChunk(
                    text=summary,
                    metadata=metadata,
                    content_type="text"  # Usa sempre "text" per uniformità
                )
                final_text_chunks.append(chunk)
    
    # --- Processa le immagini ---
    if images:
        logging.info("\n--- Inizio riassunto immagini ---")
        image_summaries = process_in_batch(image_summarize_chain, images, max_concurrency=3)
        
        # Crea TextChunk per ogni riassunto di immagine
        for i, summary in enumerate(image_summaries):
            if summary and not summary.startswith("Errore"):
                # Assicurati che i metadati contengano almeno source e page
                metadata = {
                    "source": images[i]["metadata"].get("source", source_filename),
                    "page": images[i]["metadata"].get("page", None)
                }
                chunk = TextChunk(
                    text=summary,
                    metadata=metadata,
                    content_type="text"  # Usa sempre "text" per uniformità
                )
                final_text_chunks.append(chunk)
    
    logging.info(f"Creati {len(final_text_chunks)} chunk di testo finali")
    return final_text_chunks
