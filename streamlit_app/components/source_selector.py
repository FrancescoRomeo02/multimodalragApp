# streamlit_app/components/source_selector.py
import streamlit as st
import os
from app.config import RAW_DATA_PATH

def source_selector():
    """
    Mostra una lista di documenti disponibili nella directory 'data/raw'
    e permette all'utente di selezionare su quali eseguire le query.
    Restituisce la lista dei nomi dei file selezionati.
    """
    st.header("1. Seleziona le Fonti")
    
    # Assicurati che la directory esista
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        st.info("La cartella 'data/raw' è stata creata. Carica dei documenti per iniziare.")
        return []

    # Elenca i file PDF nella directory
    available_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".pdf")]

    if not available_files:
        st.info("Nessun documento trovato. Carica un PDF usando il pannello di upload.")
        return []

    # Usa lo stato della sessione per "ricordare" quali file sono selezionati
    if 'selected_files' not in st.session_state:
        # Di default, seleziona tutti i file disponibili all'avvio
        st.session_state.selected_files = available_files

    st.write("Seleziona i documenti da includere nella tua ricerca:")
    
    selected_now = []
    for file in available_files:
        # La checkbox usa lo stato della sessione per mantenere la selezione
        is_selected = st.checkbox(
            file, 
            value=(file in st.session_state.selected_files), 
            key=f"cb_{file}" # Chiave unica per ogni checkbox
        )
        if is_selected:
            selected_now.append(file)
            
    # Aggiorna lo stato della sessione con la nuova selezione
    st.session_state.selected_files = selected_now

    if not st.session_state.selected_files:
        st.warning("Attenzione: non hai selezionato alcun documento. Le ricerche non produrranno risultati.")
    
    return st.session_state.selected_files