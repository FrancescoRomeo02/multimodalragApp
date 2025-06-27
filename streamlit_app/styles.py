# streamlit_app/styles.py

def get_custom_css():
    """Restituisce una stringa CSS per uno stile minimalista e professionale."""
    return """
    <style>
        /* Nasconde header e footer di Streamlit */
        #MainMenu {visibility: ;}
        footer {visibility: ;}
        header {visibility: ;}

        /* Stile generale del contenitore */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        /* Migliora lo stile dei widget */
        .st-emotion-cache-1y4p8pa { /* Contenitore del file uploader */
            border: 1px dashed #ced4da;
            border-radius: 0.5rem;
        }
        
        .stButton>button {
            border-radius: 0.5rem;
            width: 100%;
        }

        /* Stile per i messaggi della chat */
        .stChatMessage {
            border-radius: 0.75rem;
            border: 1px solid #e9ecef;
            padding: 1rem;
        }
    </style>
    """