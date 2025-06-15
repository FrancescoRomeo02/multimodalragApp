import sys
import os
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from components.upload import upload_and_process_file
from components.viewer import viewer

from components.chat import get_rag_response

st.set_page_config(page_title="MultimodalRAG", layout="wide")

st.title("MultimodalRAG")

# Sidebar navigation
st.sidebar.title("Navigazione")
page = st.sidebar.radio(
    "Vai a:",
    ("Carica PDF", "Gestione Risorse", "Chat", "Impostazioni")
)

if page == "Carica PDF":
    st.header("Carica un nuovo PDF")
    uploaded_file = upload_and_process_file()
    if uploaded_file:
        st.success("PDF caricato con successo!")
    else:
        st.warning("Carica un PDF per iniziare.")

elif page == "Gestione Risorse":
    st.header("Gestione PDF e Risorse")
    viewer()


elif page == "Chat":
    st.header("Chatta con il Bot")
    st.info("Qui potrai interagire con il bot per ottenere risposte dai PDF caricati.")
    query = st.text_input("Inserisci la tua domanda:")
    if query:
        response = get_rag_response(query)
        st.subheader("Risposta del Bot:")
        st.write(response['answer'])
        
        st.subheader("Fonti utilizzate:")
        for source in response['sources']:
            st.write(f"**Pagina:** {source['page']} - **Tipo:** {source['type']} - **Testo:** {source['text'][:200]}... (Score: {source['score']})")


elif page == "Visualizzatore PDF":
    st.header("Visualizzatore PDF")
    # TODO: mostra i PDF caricati e permetti la visualizzazione
    st.info("Qui potrai visualizzare i PDF caricati.")

elif page == "Utils":
    st.header("Strumenti Utili")
    # TODO: aggiungi strumenti utili (es. estrazione testo, statistiche, ecc.)
    st.info("Qui verranno aggiunti strumenti utili per la gestione dei PDF.")

st.sidebar.markdown("---")
st.sidebar.info("MultimodalRAG - Università LM 24-25")