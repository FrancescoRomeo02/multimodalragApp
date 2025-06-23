# streamlit_app/components/ui_components.py

import streamlit as st
import os
from app.config import RAW_DATA_PATH
from app.pipeline.retriever import create_rag_chain
from streamlit_app.backend_logic import process_uploaded_file, delete_source

def upload_widget(indexer):
    """Disegna il widget di upload e gestisce l'indicizzazione in modo sicuro."""
    st.header("Carica un Nuovo Documento")
    uploaded_file = st.file_uploader(
        "Scegli un file PDF", type="pdf", label_visibility="collapsed"
    )

    if uploaded_file:
        # --- INIZIO DELLA CORREZIONE ---
        
        # Creiamo una chiave unica per questo specifico file caricato.
        # Usare nome + dimensione lo rende robusto anche se si ricarica lo stesso file.
        upload_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"

        # Eseguiamo l'indicizzazione SOLO SE non abbiamo già processato questo file
        # nella sessione corrente.
        if upload_key not in st.session_state:
            with st.spinner(f"Indicizzazione di '{uploaded_file.name}' in corso..."):
                success, message = process_uploaded_file(uploaded_file, indexer)
            
            if success:
                st.success(message)
                # Marchiamo il file come processato nella sessione.
                st.session_state[upload_key] = True
                # Ora possiamo fare il rerun in sicurezza per aggiornare la UI.
                st.rerun()
            else:
                st.error(message)
                # Marchiamo anche i fallimenti per non riprovare all'infinito.
                st.session_state[upload_key] = True
        
        # --- FINE DELLA CORREZIONE ---


def source_selector_widget():
    """Disegna il widget per selezionare ed eliminare le fonti."""
    st.header("Gestisci e Seleziona le Fonti")
    
    if not os.path.exists(RAW_DATA_PATH) or not os.listdir(RAW_DATA_PATH):
        st.info("Nessun documento trovato. Caricane uno per iniziare.")
        return []

    available_files = sorted([f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".pdf")])
    
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = available_files

    selected_now = st.multiselect(
        "Seleziona i documenti per la chat:", 
        options=available_files, 
        default=st.session_state.selected_files
    )
    st.session_state.selected_files = selected_now

    st.markdown("---")
    with st.expander("🗑️ Elimina documenti"):
        for file in available_files:
            if st.button(f"Elimina '{file}'", key=f"del_{file}"):
                if file in st.session_state.selected_files:
                    st.session_state.selected_files.remove(file)
                with st.spinner(f"Eliminazione di '{file}' in corso..."):
                    success, message = delete_source(file)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    return selected_now

def chat_interface_widget(selected_sources: list[str]):
    """Disegna l'interfaccia di chat."""
    st.header("Chatta con le Fonti Selezionate")

    if not selected_sources:
        st.warning("Seleziona almeno una fonte per avviare la chat.")
        return

    # Logica per creare/caricare la catena RAG
    if 'rag_chain' not in st.session_state or st.session_state.get('last_selected_sources') != selected_sources:
        with st.spinner(f"Preparazione del motore di ricerca per {len(selected_sources)} fonti..."):
            st.session_state.rag_chain = create_rag_chain(selected_sources)
        st.session_state.last_selected_sources = selected_sources

    rag_chain = st.session_state.rag_chain

    # Inizializzazione e visualizzazione messaggi
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ciao! Fai una domanda sulle fonti selezionate."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input utente e risposta del bot
    if prompt := st.chat_input("Fai la tua domanda..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Sto pensando..."):
                response = rag_chain.invoke({"query": prompt})
                answer = response.get("result", "Nessuna risposta trovata.")
                sources = response.get("source_documents", [])
                st.markdown(answer)
                
                assistant_message = {"role": "assistant", "content": answer, "sources": sources}
                st.session_state.messages.append(assistant_message)
                st.rerun()