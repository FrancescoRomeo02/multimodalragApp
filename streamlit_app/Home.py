import sys
import os
import streamlit as st

# --- Bootstrap per risolvere gli import ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- Fine Bootstrap ---

# Importa i tuoi componenti
from streamlit_app.components.upload import upload_and_process_file
from streamlit_app.components.chat import chat_interface
from streamlit_app.components.source_selector import source_selector

# Configurazione della pagina
st.set_page_config(
    layout="wide",
    page_title="RAG per Paper Scientifici",
    page_icon="🔬"
)

st.title("Assistente di Ricerca per Paper Scientifici 🔬")
st.markdown("Carica un PDF, attendi l'indicizzazione e poi poni le tue domande.")

# Layout a due colonne
col1, col2 = st.columns([0.4, 0.6]) # Colonne di uguale dimensione

with col1:
    selected_sources = source_selector()
    # Componente di upload
    upload_and_process_file()

with col2:
    # Componente della chat
    chat_interface(selected_sources=selected_sources)