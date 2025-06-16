import streamlit as st
from app.pipeline.retriever import create_rag_chain

def chat_interface(selected_sources: list[str]):
    """
    Crea e gestisce l'interfaccia di chat completa in Streamlit.
    """
    st.header("2. Chatta con le fonti selezionate")

    if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Ciao! Seleziona le fonti e fai la tua domanda."}]

    # Inizializza la catena RAG in session_state per evitare di ricaricarla
    # se i file selezionati non sono cambiati.
    if 'rag_chain' not in st.session_state or st.session_state.get('last_selected_sources') != selected_sources:
        if not selected_sources:
            st.session_state.rag_chain = None
        else:
            with st.spinner(f"Inizializzazione del motore di ricerca per {len(selected_sources)} documenti..."):
                st.session_state.rag_chain = create_rag_chain(selected_sources)
        st.session_state.last_selected_sources = selected_sources
    
    rag_chain = st.session_state.rag_chain

    if rag_chain is None:
        st.stop() # Interrompe l'esecuzione se la catena non può essere caricata

    # Inizializza la cronologia della chat in st.session_state se non esiste
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ciao! Sono il tuo assistente di ricerca. Fai una domanda sul documento che hai caricato."}
        ]

    # Mostra i messaggi precedenti
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Se il messaggio è dell'assistente e ha delle fonti, mostrale
            if "sources" in message:
                with st.expander("Vedi Fonti Utilizzate"):
                    for i, source in enumerate(message["sources"]):
                        st.info(
                            f"**Fonte {i+1}** (Pagina: {source.metadata.get('page_number', 'N/A')})\n\n" # type: ignore
                            f"```{source.page_content}```" # type: ignore
                        )

    # Input dell'utente
    if prompt := st.chat_input("Fai la tua domanda..."):
        # Aggiungi il messaggio dell'utente alla cronologia e mostralo
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Mostra la risposta dell'assistente
        with st.chat_message("assistant"):
            # Aggiungi un placeholder per l'effetto "sto pensando..."
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Ricerca nelle fonti selezionate...")

            # Esegui la query RAG
            try:
                response = rag_chain.invoke({"query": prompt})
                
                answer = response.get("result", "Non ho trovato una risposta.")
                source_documents = response.get("source_documents", [])
                
                # Aggiorna il placeholder con la risposta reale
                message_placeholder.markdown(answer)
                
                # Salva il messaggio completo dell'assistente nella sessione
                assistant_message = {"role": "assistant", "content": answer, "sources": source_documents}
                st.session_state.messages.append(assistant_message)

                # Mostra le fonti in un expander sotto la risposta
                with st.expander("Vedi Fonti Utilizzate"):
                    for i, source in enumerate(source_documents):
                        st.info(
                            f"**Fonte {i+1}** (Pagina: {source.metadata.get('page_number', 'N/A')})\n\n"
                            f"```{source.page_content}```"
                        )

            except Exception as e:
                error_message = f"Si è verificato un errore: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})