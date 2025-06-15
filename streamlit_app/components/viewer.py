import os
import shutil
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import pandas as pd

RAW_DIR = "data/raw"
PDFS_IMAGES_DIR = "pdfs_images"
PDFS_TABLES_DIR = "pdfs_tables"
PDFS_TEXT_DIR = "pdfs_text"
SELECTION_FILE = "selected_resources.txt"

@st.dialog("📄 Visualizzatore PDF", width="large")
def show_pdf_dialog(pdf_path):
    try:
        pdf_viewer(pdf_path, width=750, height=950, annotations=[])
    except Exception as e:
        st.error(f"Errore nella visualizzazione del PDF: {e}")

# 📂 Utility per la selezione
def load_selection():
    files = list_files(RAW_DIR, [".pdf", ".png", ".jpg", ".jpeg"])
    existing_selection = set()
    if os.path.exists(SELECTION_FILE):
        with open(SELECTION_FILE, "r") as f:
            existing_selection = set(line.strip() for line in f)
    else:
        # Se non esiste, inizializza tutto come selezionato
        with open(SELECTION_FILE, "w") as f:
            for file in files:
                f.write(f"{os.path.join(RAW_DIR, file)}\n")
        existing_selection = set(os.path.join(RAW_DIR, f) for f in files)
    return existing_selection

def save_selection(selection):
    with open(SELECTION_FILE, "w") as f:
        for item in selection:
            f.write(f"{item}\n")

def list_files(directory, extensions=None):
    files = []
    for fname in os.listdir(directory):
        if not extensions or any(fname.lower().endswith(ext) for ext in extensions):
            files.append(fname)
    return files


# 🔁 Componente riutilizzabile
def file_card(label, path, key_prefix, show_image=False, show_table=False):
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        if show_image:
            st.image(path, width=300, caption=os.path.basename(path))
        elif show_table:
            try:
                if path.endswith(".csv"):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)
                st.markdown(f"**{os.path.basename(path)}**")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Errore nella lettura della tabella: {e}")
        else:
            st.markdown(f"**{label}**")

    with col2:
        selected = st.checkbox("Seleziona", key=f"{key_prefix}_sel", value=path in selection)
        if selected:
            updated_selection.add(path)
        else:
            updated_selection.discard(path)

    with col3:
        if st.button("Elimina", key=f"{key_prefix}_del"):
            os.remove(path)
            updated_selection.discard(path)
            st.rerun()


# 👁️ Funzione principale
def viewer():
    global selection, updated_selection
    selection = load_selection()
    updated_selection = set(selection)

    st.header("📂 PDFs e Immagini")
    pdfs = sorted(list_files(RAW_DIR, [".pdf"]))
    images = sorted(list_files(RAW_DIR, [".png", ".jpg", ".jpeg"]))

    for pdf in pdfs:
        st.markdown("----")
        st.markdown(f"### 📄 {pdf}")
        full_pdf_path = os.path.join(RAW_DIR, pdf)

        # Checkbox di selezione
        selected = st.checkbox("Seleziona", key=f"pdf_{pdf}", value=full_pdf_path in selection)
        if selected:
            updated_selection.add(full_pdf_path)
        else:
            updated_selection.discard(full_pdf_path)

        # Pulsanti
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Elimina", key=f"del_pdf_{pdf}"):
                shutil.rmtree(os.path.join(RAW_DIR, PDFS_IMAGES_DIR, pdf.replace(".pdf", "")), ignore_errors=True)
                shutil.rmtree(os.path.join(RAW_DIR, PDFS_TABLES_DIR, pdf.replace(".pdf", "")), ignore_errors=True)
                os.remove(os.path.join(RAW_DIR, PDFS_TEXT_DIR, pdf.replace(".pdf", ".txt")))
                os.remove(full_pdf_path)
                updated_selection.discard(full_pdf_path)
                st.rerun()
        with col2:
            if st.button("Visualizza", key=f"view_pdf_{pdf}"):
                show_pdf_dialog(full_pdf_path)

        # 📸 Immagini associate
        pdf_images_dir = os.path.join(RAW_DIR, PDFS_IMAGES_DIR, pdf.replace(".pdf", ""))
        if os.path.exists(pdf_images_dir) and list_files(pdf_images_dir, [".png", ".jpg", ".jpeg"]):
            st.subheader("🖼️ Immagini Estratte")
            for img_file in sorted(list_files(pdf_images_dir, [".png", ".jpg", ".jpeg"])):
                img_path = os.path.join(pdf_images_dir, img_file)
                file_card(img_file, img_path, f"img_{img_file}", show_image=True)
                st.markdown("---")

        # 📊 Tabelle associate
        pdf_tables_dir = os.path.join(RAW_DIR, PDFS_TABLES_DIR, pdf.replace(".pdf", ""))
        if os.path.exists(pdf_tables_dir) and list_files(pdf_tables_dir, [".csv", ".xlsx"]):
            st.subheader("📊 Tabelle Estratte")
            for table_file in sorted(list_files(pdf_tables_dir, [".csv", ".xlsx"])):
                table_path = os.path.join(pdf_tables_dir, table_file)
                file_card(table_file, table_path, f"table_{table_file}", show_table=True)
                st.markdown("---")
        

    # 📷 Immagini singole non associate a PDF
    if images:
        st.header("🖼️ Immagini Singole")
        for img in images:
            st.markdown("----")
            img_path = os.path.join(RAW_DIR, img)
            file_card(img, img_path, f"img_single_{img}", show_image=True)

    # 💾 Salvataggio
    if updated_selection != selection:
        save_selection(updated_selection)
        st.success("Selezione aggiornata!")