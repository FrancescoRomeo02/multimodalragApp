import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from PIL import Image
import os

# Importa le configurazioni e la pipeline di indicizzazione
from app.config import RAW_DATA_PATH
from app.pipeline.indexer import index_document

def upload_and_process_file():
    """
    Gestisce l'upload di file (PDF o immagini) da parte dell'utente,
    li salva e avvia la pipeline di indicizzazione per i PDF.
    """
    st.header("1. Carica un Nuovo Documento")
    st.write("Carica un paper scientifico in formato PDF per analizzarlo e porre domande.")

    uploaded_file = st.file_uploader(
        "Scegli un file PDF o un'immagine",
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Crea la directory 'raw' se non esiste
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        
        # Percorso dove salvare il file originale
        save_path = os.path.join(RAW_DATA_PATH, uploaded_file.name)

        # Salva il file caricato
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' salvato con successo in `data/raw/`.")

        file_type = uploaded_file.type

        # --- Gestione dei PDF ---
        if file_type == "application/pdf":
            # Avvia la pipeline di indicizzazione
            st.info("Avvio della pipeline di indicizzazione... Questo processo analizzerà la struttura, creerà i chunk e li vettorizzerà.")
            
            with st.spinner(f"Indicizzazione di '{uploaded_file.name}' in corso..."):
                try:
                    # Questa è la chiamata chiave alla nostra nuova pipeline!
                    index_document(save_path)
                    st.success("Indicizzazione completata! ✅ Ora puoi fare domande sul documento nella chat.")
                except Exception as e:
                    st.error(f"Si è verificato un errore durante l'indicizzazione: {e}")
                    st.error("Assicurati che Qdrant sia in esecuzione e che le dipendenze (tesseract, poppler) siano installate.")
            
            # Mostra il PDF all'utente per conferma visiva
            st.subheader("Visualizzatore PDF")
            pdf_viewer(save_path, annotations=[])

        # --- Gestione delle Immagini ---
        elif file_type.startswith("image/"):
            # Per ora, mostriamo solo l'immagine. L'indicizzazione multimodale
            # sarà un passo successivo.
            st.info("Immagine caricata. L'analisi di immagini (RAG multimodale) può essere implementata come estensione futura.")
            
            image = Image.open(save_path)
            st.image(image, caption="Immagine Caricata", use_container_width=True)

        else:
            st.error("Tipo di file non supportato.")