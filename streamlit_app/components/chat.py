import streamlit as st
from app.pipeline.retriever import create_rag_chain

@st.cache_resource
def load_rag_chain():
    """
    Carica la catena RAG e la mette in cache per evitare di ricaricarla
    a ogni interazione dell'utente. La cache viene invalidata solo se
    il codice della funzione cambia o l'app viene riavviata.
    """
    try:
        return create_rag_chain()
    except Exception as e:
        st.error(f"Errore durante il caricamento del sistema RAG: {e}")
        st.warning("Assicurati che le chiavi API (JINA, GROQ) siano nel tuo file .env e che Qdrant sia in esecuzione.")
        return None

def chat_interface():
    """
    Crea e gestisce l'interfaccia di chat completa in Streamlit.
    """
    st.header("2. Chatta con i tuoi documenti")

    # Carica la catena RAG usando la cache di Streamlit
    rag_chain = load_rag_chain()

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
                            f"**Fonte {i+1}** (Pagina: {source.metadata.get('page_number', 'N/A')})\n\n"
                            f"```{source.page_content}```"
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
            message_placeholder.markdown("⏳ Sto pensando...")

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