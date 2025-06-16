# app/utils/pdf_parser.py
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element

def parse_pdf_elements(pdf_path: str) -> List[Element]:
    """
    Usa 'unstructured' per fare il parsing di un PDF in elementi granulari.
    Questa è la funzione più potente per ottenere il massimo dettaglio.
    """
    print(f"[*] Parsing granulare del PDF: {pdf_path}")
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=False,
    )
    print(f"[*] Trovati {len(elements)} elementi strutturali (paragrafi, tabelle, ecc.).")
    return elements