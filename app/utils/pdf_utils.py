# multimodalrag/app/utils/pdf_utils.py

from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredFileLoader

def parse_pdf_elements(pdf_path: str) -> List[Document]:
    """
    Analizza un file PDF utilizzando unstructured per estrarre elementi logici.
    
    Questo approccio è "layout-aware" e preserva la struttura di tabelle,
    titoli e paragrafi.

    Args:
        pdf_path (str): Il percorso del file PDF da analizzare.

    Returns:
        List[Document]: Una lista di oggetti Document, dove ogni documento
                        rappresenta un elemento logico del PDF (es. un paragrafo,
                        una tabella in HTML, un titolo).
    """
    print(f"[*] Analisi strutturale del PDF: {pdf_path}")
    
    loader = UnstructuredFileLoader(
        file_path=pdf_path,
        strategy="hi_res",
        mode="elements",
        unstructured_kwargs={
            "pdf_infer_table_structure": True,
            "extract_images_in_pdf": False,
        }
    )
    
    elements = loader.load()
    print(f"[*] Trovati {len(elements)} elementi strutturali nel documento.")
    return elements
