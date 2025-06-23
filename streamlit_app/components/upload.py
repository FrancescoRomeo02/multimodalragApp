# file: app_ui.py (o il tuo file principale di Streamlit)

import streamlit as st
import os
import base64
import uuid
import logging

# --- Importazioni dai moduli della tua applicazione ---
from app.config import RAW_DATA_PATH, QDRANT_URL, COLLECTION_NAME
from app.pipeline.indexer_service import DocumentIndexer
from app.utils.embedder import get_multimodal_embedding_model

# Configura il logging per una migliore diagnostica
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Correzione della Funzione di Caching ---
# Questa funzione ora è "pulita": non contiene elementi della UI.
# Si occupa solo di caricare le risorse.
@st.cache_resource
def load_services() -> DocumentIndexer:
    """
    Carica e inizializza i servizi pesanti (Embedder e Indexer) una sola volta.
    Se il caricamento fallisce, solleva un'eccezione che verrà gestita
    dalla UI principale.
    """
    logger.info("Inizializzazione dei servizi in corso (questa operazione è cachata)...")
    embedder = get_multimodal_embedding_model()  # Questo è il passo lento, eseguito una sola volta.
    indexer = DocumentIndexer(
        embedder=embedder,
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION_NAME,

    )
    logger.info("Servizi inizializzati e pronti.")
    return indexer

def handle_image_upload(file_path: str, file_name: str, indexer: DocumentIndexer):
    """
    Logica specifica per l'indicizzazione di un singolo file immagine.
    """
    st.info("Avvio indicizzazione per l'immagine...")
    with st.spinner(f"Vettorizzazione di '{file_name}' in corso..."):
        try:
            with open(file_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            image_vector_result = indexer.embedder.embed_image_from_base64(
                img_base64,
                description=f"Immagine utente: {file_name}"
            )
            # Se image_vector_result è una tupla (vector, metadata), estrai solo il vettore
            if isinstance(image_vector_result, tuple):
                image_vector, _ = image_vector_result
            else:
                image_vector = image_vector_result

            metadata = {
                "source": file_name,
                "type": "image",
                "description": f"Immagine utente: {file_name}"
            }
            
            from qdrant_client.http.models import PointStruct

            point = PointStruct(
                id=str(uuid.uuid4()),  # Genera un ID unico per il punto
                vector=image_vector,
                payload={
                    "metadata": metadata,
                    "file_name": file_name
                }
            )
            indexer.qdrant_client.upsert(
                collection_name=indexer.collection_name,
                points=[point],
                wait=True
            )
            st.success("Indicizzazione immagine completata! ✅")
        except Exception as e:
            st.error(f"Errore durante l'indicizzazione dell'immagine: {e}")

def main():
    """
    Funzione principale che disegna l'interfaccia utente di Streamlit.
    """

    # --- Gestione UI del Caricamento Servizi ---
    # Qui gestiamo la UI, separata dalla logica di caching.
    try:
        with st.spinner("Caricamento del modello di embedding e connessione al database... 🧠"):
            indexer = load_services()
        st.toast("Modello caricato e pronto! ✅")
    except Exception as e:
        logger.error(f"Errore critico durante l'inizializzazione dei servizi: {e}", exc_info=True)
        st.error(f"Errore critico durante l'inizializzazione dei servizi: {e}")
        st.error("L'applicazione non può continuare. Controlla i log, la connessione a Qdrant e assicurati che il modello di embedding sia accessibile.")
        st.stop() # Interrompe l'esecuzione dello script se i servizi non possono essere caricati.

    st.header("Carica un Nuovo Documento")
    
    uploaded_file = st.file_uploader(
        "Scegli un file PDF o un'immagine (PNG, JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        save_path = os.path.join(RAW_DATA_PATH, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' salvato in `data/raw/`.")

        file_type = uploaded_file.type

        if file_type == "application/pdf":
            st.info("Avvio della pipeline di indicizzazione per il PDF...")
            with st.spinner(f"Indicizzazione di '{uploaded_file.name}' in corso..."):
                try:
                    indexer.index_files([save_path], force_recreate=False)
                    st.success("Indicizzazione PDF completata! ✅")
                except Exception as e:
                    logger.error(f"Errore durante l'indicizzazione del PDF: {e}", exc_info=True)
                    st.error(f"Errore durante l'indicizzazione del PDF: {e}")
            

        elif file_type.startswith("image/"):
            handle_image_upload(save_path, uploaded_file.name, indexer)

        else:
            st.error("Tipo di file non supportato.")
