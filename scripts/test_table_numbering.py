#!/usr/bin/env python3
"""
Test specifico per verificare che la numerazione delle tabelle funzioni
con un PDF di esempio che contiene tabelle
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
from src.utils.pdf_parser import parse_pdf_elements

def create_test_pdf_with_tables():
    """Crea un PDF di test semplice con tabelle"""
    doc = fitz.open()
    page = doc.new_page()
    
    # Aggiungi testo e una tabella semplice
    text = """
    Documento di Test con Tabelle
    
    Questo è un documento creato per testare l'estrazione di tabelle.
    
    Nome    Età    Città
    Marco   30     Roma
    Laura   25     Milano
    Paolo   35     Napoli
    
    Ecco una seconda tabella:
    
    Prodotto    Prezzo    Quantità
    Mela        1.50      10
    Pera        2.00      5
    Banana      1.20      15
    """
    
    page.insert_text((50, 50), text)
    
    # Salva il PDF temporaneo
    temp_pdf = "temp_test_tables.pdf"
    doc.save(temp_pdf)
    doc.close()
    return temp_pdf

def test_table_numbering():
    """Test della numerazione delle tabelle"""
    print("=== TEST NUMERAZIONE TABELLE ===\n")
    
    # Crea PDF di test
    temp_pdf = create_test_pdf_with_tables()
    
    try:
        # Prova prima con estrazione diretta delle tabelle
        print("🔍 Test estrazione diretta tabelle...")
        doc = fitz.open(temp_pdf)
        page = doc.load_page(0)
        
        from src.utils.pdf_parser import extract_tables_from_page
        tables_direct = extract_tables_from_page(page)
        print(f"Tabelle estratte direttamente: {len(tables_direct)}")
        
        if tables_direct:
            for i, table in enumerate(tables_direct):
                print(f"  Tabella {i+1}: {table['table_data']['shape']}")
                preview = table['table_markdown'][:100].replace('\n', ' ')
                print(f"    Preview: {preview}...")
        
        doc.close()
        
        # Test completo con parse_pdf_elements
        print("\n🔍 Test completo con numerazione...")
        text_elements, image_elements, table_elements = parse_pdf_elements(temp_pdf)
        
        print(f"Risultati: {len(text_elements)} testi, {len(image_elements)} immagini, {len(table_elements)} tabelle")
        
        if table_elements:
            print("\n📋 ANALISI NUMERAZIONE TABELLE:")
            for i, table in enumerate(table_elements):
                table_number = table['metadata'].get('table_number', 'N/A')
                page = table['metadata'].get('page', 'N/A')
                content = table['table_markdown']
                has_number = f"Tabella: {table_number}" in content
                
                print(f"  Tabella {i+1}:")
                print(f"    • Numero globale: {table_number}")
                print(f"    • Pagina: {page}")
                print(f"    • Numero nell'embedding: {'✅' if has_number else '❌'}")
                preview = content[:80].replace('\n', ' ')
                print(f"    • Preview: {preview}...")
                print()
        else:
            print("⚠️ Nessuna tabella estratta con numerazione")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Pulisci file temporaneo
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)
            print(f"🗑️ File temporaneo {temp_pdf} rimosso")

if __name__ == "__main__":
    test_table_numbering()
