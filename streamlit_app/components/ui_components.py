# streamlit_app/components/ui_components.py

import streamlit as st
import os
import base64
from io import BytesIO
from PIL import Image
from app.config import RAW_DATA_PATH
from app.pipeline.retriever import edit_answer, enhanced_rag_query
from streamlit_app.backend_logic import process_uploaded_file, delete_source

def upload_widget(indexer):
    """Disegna il widget di upload e gestisce l'indicizzazione in modo sicuro."""
    st.header("Carica un Nuovo Documento")
    uploaded_file = st.file_uploader(
        "Scegli un file PDF", type="pdf", label_visibility="collapsed"
    )

    if uploaded_file:
        # Creiamo una chiave unica per questo specifico file caricato.
        upload_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"

        # Eseguiamo l'indicizzazione SOLO SE non abbiamo già processato questo file
        if upload_key not in st.session_state:
            with st.spinner(f"Indicizzazione di '{uploaded_file.name}' in corso..."):
                success, message = process_uploaded_file(uploaded_file, indexer)
            
            if success:
                st.success(message)
                st.session_state[upload_key] = True
                st.rerun()
            else:
                st.error(message)
                st.session_state[upload_key] = True


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


def display_source_document(doc_info: dict, index: int):
    """
    Visualizza le informazioni di un documento di origine in modo più preciso e pulito.
    
    Args:
        doc_info: Dizionario con le informazioni del documento
        index: Indice del documento
    """
    doc_type = doc_info.get("type", "text")
    source = doc_info.get("source", "Sconosciuto")
    page = doc_info.get("page", "N/A")
    
    # Emoji e colori per tipo di documento
    type_info = {
        "text": {"emoji": "📝", "color": "#1f77b4"},
        "image": {"emoji": "🖼️", "color": "#2ca02c"}, 
        "table": {"emoji": "📊", "color": "#ff7f0e"}
    }
    
    info = type_info.get(doc_type, {"emoji": "📄", "color": "#7f7f7f"})
    
    # Header compatto e preciso
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, {info["color"]}22 0%, {info["color"]}11 100%);
        border-left: 4px solid {info["color"]};
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.9em;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <strong>{info["emoji"]} {source}</strong> - Pagina {page}
            </div>
            <span style="
                background-color: {info["color"]};
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.75em;
                font-weight: bold;
            ">{doc_type.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Contenuto specifico per tipo in expander compatto
    with st.expander(f"👁️ Dettagli del documento {index+1}", expanded=False):
        
        if doc_type == "table":
            # Informazioni tabella
            caption = doc_info.get("caption")
            context = doc_info.get("context_text")
            
            if caption and len(caption.strip()) > 0:
                st.info(f"📋 **Caption:** {caption}")
            
            if context and len(context.strip()) > 0:
                with st.expander("🔍 Contesto circostante"):
                    st.text(context)
            
            # Contenuto tabella con miglior formatting
            table_content = doc_info.get("table_markdown_raw") or doc_info.get("content", "")
            if table_content:
                st.markdown("**📊 Contenuto della tabella:**")
                st.markdown(table_content)
            
        elif doc_type == "image":
            # Informazioni immagine
            manual_caption = doc_info.get("manual_caption")
            image_caption = doc_info.get("image_caption")
            context = doc_info.get("context_text")
            
            # Caption più compatte
            if manual_caption and len(manual_caption.strip()) > 0:
                st.success(f"👨‍💻 **Caption manuale:** {manual_caption}")
            
            if image_caption and len(image_caption.strip()) > 0:
                st.info(f"🤖 **Caption AI:** {image_caption}")
            
            if context and len(context.strip()) > 0:
                with st.expander("🔍 Contesto circostante"):
                    st.text(context)
            
            # Immagine con dimensioni ottimizzate
            image_base64 = doc_info.get("image_base64")
            if image_base64:
                try:
                    image_data = base64.b64decode(image_base64)
                    image = Image.open(BytesIO(image_data))
                    
                    # Ridimensiona l'immagine se troppo grande
                    max_width = 400
                    if image.width > max_width:
                        ratio = max_width / image.width
                        new_height = int(image.height * ratio)
                        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
                    
                    st.image(image, caption=f"Da {source} - Pagina {page}")
                except Exception as e:
                    st.error(f"❌ Errore nel caricamento dell'immagine: {e}")
            
        else:  # text
            # Contenuto testuale con anteprima intelligente
            content = doc_info.get("content", "")
            if len(content) > 200:
                preview = content[:200] + "..."
                st.text_area("📝 Anteprima contenuto", preview, height=100, disabled=True)
                
                if st.button(f"📖 Mostra tutto", key=f"expand_text_{index}"):
                    st.text_area("📝 Contenuto completo", content, height=300, disabled=True)
            else:
                st.text_area("📝 Contenuto", content, height=100, disabled=True)


def enhanced_chat_interface_widget(selected_sources: list[str]):
    """
    Interfaccia di chat migliorata con riferimenti precisi e puliti
    """
    st.header("🤖 Chatta con le Fonti Selezionate")

    if not selected_sources:
        st.warning("⚠️ Seleziona almeno una fonte per avviare la chat.")
        return

    # Opzioni sempre attive (rimosse dalla UI per semplicità)
    show_sources = True
    multimodal = True
    compact_sources = True

    # Inizializzazione messaggi
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ciao! Fai una domanda sui documenti selezionati. I riferimenti precisi saranno mostrati sotto ogni risposta."}
        ]

    # Visualizzazione messaggi
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Mostra documenti di origine in modo migliorato
            if (msg["role"] == "assistant" and 
                "source_docs" in msg and 
                show_sources and 
                msg["source_docs"]):
                
                st.markdown("---")
                
                if compact_sources:
                    # Visualizzazione compatta con riferimenti precisi
                    st.markdown("**📚 Documenti di origine:**")
                    
                    # Raggruppa per fonte e pagina
                    sources_summary = {}
                    for doc in msg["source_docs"]:
                        source = doc.get("source", "Sconosciuto")
                        page = doc.get("page", "N/A")
                        doc_type = doc.get("type", "text")
                        
                        key = f"{source}"
                        if key not in sources_summary:
                            sources_summary[key] = {"pages": set(), "types": set()}
                        
                        sources_summary[key]["pages"].add(str(page))
                        sources_summary[key]["types"].add(doc_type)
                    
                    # Mostra riferimenti in formato compatto e preciso
                    for source, info in sources_summary.items():
                        pages_list = sorted(info["pages"], key=lambda x: int(x) if x.isdigit() else float('inf'))
                        types_list = sorted(info["types"])
                        
                        # Formato più compatto: "📄 documento.pdf: p.1,2,3 (text, image)"
                        if len(pages_list) == 1:
                            pages_str = f"p.{pages_list[0]}"
                        else:
                            pages_str = f"pp.{','.join(pages_list)}"
                        
                        types_str = ", ".join(types_list)
                        
                        # Emoji per tipo di documento
                        if "image" in types_list:
                            emoji = "🖼️"
                        elif "table" in types_list:
                            emoji = "📊"
                        else:
                            emoji = "📄"
                        
                        st.markdown(f"{emoji} **{source}**: {pages_str} *({types_str})*")
                
                else:
                    # Visualizzazione dettagliata
                    st.markdown("**📚 Documenti di origine dettagliati:**")
                    
                    for i, doc in enumerate(msg["source_docs"]):
                        display_source_document(doc, i)

    # Input utente e generazione risposta
    if prompt := st.chat_input("Fai la tua domanda..."):
        # Aggiungi messaggio utente
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Genera risposta
        with st.chat_message("assistant"):
            with st.spinner("🔍 Sto analizzando i documenti..."):
                response = enhanced_rag_query(
                    query=prompt,
                    selected_files=selected_sources,
                    multimodal=multimodal
                )
                
                if not response:
                    answer = "❌ Spiacente, non sono riuscito a trovare informazioni rilevanti nei documenti selezionati."
                    source_docs = []
                else:
                    answer = response.answer or "❌ Nessuna risposta generata."
                    source_docs = response.source_documents or []

                # Modifica la risposta per migliorare la qualità
                edited_answer = edit_answer(answer).content
                st.markdown(edited_answer)

                # Salva messaggio con documenti di origine
                assistant_message = {
                    "role": "assistant", 
                    "content": edited_answer,
                    "source_docs": source_docs if show_sources else []
                }
                st.session_state.messages.append(assistant_message)
                
                # Mostra riferimenti immediati
                if show_sources and source_docs:
                    st.markdown("---")
                    
                    if compact_sources:
                        # Riferimenti compatti e precisi
                        st.markdown("**📚 Documenti di origine:**")
                        
                        sources_summary = {}
                        for doc in source_docs:
                            source = doc.get("source", "Sconosciuto")
                            page = doc.get("page", "N/A")
                            doc_type = doc.get("type", "text")
                            
                            key = f"{source}"
                            if key not in sources_summary:
                                sources_summary[key] = {"pages": set(), "types": set()}
                            
                            sources_summary[key]["pages"].add(str(page))
                            sources_summary[key]["types"].add(doc_type)
                        
                        # Mostra riferimenti in formato compatto e preciso
                        for source, info in sources_summary.items():
                            pages_list = sorted(info["pages"], key=lambda x: int(x) if x.isdigit() else float('inf'))
                            types_list = sorted(info["types"])
                            
                            # Formato più compatto: "📄 documento.pdf: p.1,2,3 (text, image)"
                            if len(pages_list) == 1:
                                pages_str = f"p.{pages_list[0]}"
                            else:
                                pages_str = f"pp.{','.join(pages_list)}"
                            
                            types_str = ", ".join(types_list)
                            
                            # Emoji per tipo di documento
                            if "image" in types_list:
                                emoji = "🖼️"
                            elif "table" in types_list:
                                emoji = "📊"
                            else:
                                emoji = "📄"
                            
                            st.markdown(f"{emoji} **{source}**: {pages_str} *({types_str})*")
                    
                    else:
                        # Riferimenti dettagliati
                        st.markdown("**📚 Documenti di origine dettagliati:**")
                        for i, doc in enumerate(source_docs):
                            display_source_document(doc, i)
                
                st.rerun()


# Funzioni legacy per compatibilità
def chat_interface_widget(selected_sources: list[str]):
    """Wrapper per compatibilità con codice esistente."""
    return enhanced_chat_interface_widget(selected_sources)