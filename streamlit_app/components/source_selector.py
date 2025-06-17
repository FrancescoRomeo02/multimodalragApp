import streamlit as st
import os
from app.config import RAW_DATA_PATH
# Importa la nostra nuova funzione di utility
from app.utils.qdrant_utils import delete_documents_by_filename

def source_selector():
    """
    Mostra una lista di documenti, permette la selezione e l'eliminazione.
    Restituisce la lista dei nomi dei file selezionati.
    """
    st.header("1. Gestisci e Seleziona le Fonti")
    
    # ... (la parte per creare la directory e controllare se è vuota rimane identica) ...
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        st.info("La cartella 'data/raw' è stata creata. Carica dei documenti per iniziare.")
        return []

    available_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".pdf")]

    if not available_files:
        st.info("Nessun documento trovato. Carica un PDF usando il pannello di upload.")
        return []

    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = available_files

    st.write("Seleziona i documenti da includere nella tua ricerca:")
    
    # Usiamo le colonne per allineare checkbox e pulsante
    col1, col2 = st.columns([0.8, 0.2])

    selected_now = []
    # Iteriamo su una copia della lista per poterla modificare durante il ciclo
    for file in list(available_files):
        with col1:
            is_selected = st.checkbox(
                file, 
                value=(file in st.session_state.selected_files), 
                key=f"cb_{file}"
            )
            if is_selected:
                selected_now.append(file)
        
        with col2:
            # Creiamo un pulsante di eliminazione per ogni file
            if st.button("🗑️ Elimina", key=f"del_{file}", help=f"Elimina definitivamente '{file}'"):
                handle_delete(file)
                # Forza un re-run completo di Streamlit per aggiornare la lista dei file
                st.rerun()

    st.session_state.selected_files = selected_now

    if not st.session_state.selected_files:
        st.warning("Attenzione: non hai selezionato alcun documento.")
    
    return st.session_state.selected_files

def handle_delete(filename_to_delete: str):
    """
    Funzione di logica che gestisce il processo di eliminazione completo.
    """
    st.info(f"Avvio eliminazione di '{filename_to_delete}'...")

    # 1. Eliminazione da Qdrant
    with st.spinner("Eliminazione dei dati dal database vettoriale..."):
        success_qdrant, message_qdrant = delete_documents_by_filename(filename_to_delete)
    
    if success_qdrant:
        st.success(f"Dati di '{filename_to_delete}' eliminati da Qdrant.")
    else:
        st.error(f"Errore durante l'eliminazione da Qdrant: {message_qdrant}")
        # Potremmo decidere di interrompere qui per non lasciare uno stato inconsistente
        return 

    # 2. Eliminazione dal file system locale
    try:
        file_path = os.path.join(RAW_DATA_PATH, filename_to_delete)
        if os.path.exists(file_path):
            os.remove(file_path)
            st.success(f"File '{filename_to_delete}' eliminato dal disco locale.")
        else:
            st.warning(f"File '{filename_to_delete}' non trovato sul disco per l'eliminazione.")
    except Exception as e:
        st.error(f"Errore durante l'eliminazione del file locale: {e}")
        return

    # 3. Rimuovi il file dalla selezione corrente in session_state
    if filename_to_delete in st.session_state.get('selected_files', []):
        st.session_state.selected_files.remove(filename_to_delete)

    st.success(f"Eliminazione di '{filename_to_delete}' completata.")