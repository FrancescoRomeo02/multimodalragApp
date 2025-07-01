#!/usr/bin/env python3
"""
Script di test per verificare la numerazione globale di immagini e tabelle negli embedding
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.pdf_parser import parse_pdf_elements

def test_global_numbering():
    """Test della numerazione globale per immagini e tabelle"""
    
    print("=== TEST NUMERAZIONE GLOBALE IMMAGINI E TABELLE ===\n")
    
    # Test con PDF disponibile
    pdf_path = "data/raw/rag_pars.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ File PDF non trovato: {pdf_path}")
        return
    
    try:
        text_elements, image_elements, table_elements = parse_pdf_elements(pdf_path)
        
        print(f"📖 PDF analizzato: {os.path.basename(pdf_path)}")
        print(f"📊 Risultati totali:")
        print(f"  • Testi: {len(text_elements)}")
        print(f"  • Immagini: {len(image_elements)}")
        print(f"  • Tabelle: {len(table_elements)}")
        print()
        
        # Test TABELLE - verifica numerazione
        print("🔍 ANALISI TABELLE:")
        if table_elements:
            for i, table in enumerate(table_elements):
                table_number = table['metadata'].get('table_number', 'N/A')
                page = table['metadata'].get('page', 'N/A')
                
                # Verifica che il numero sia presente nel contenuto per l'embedding
                table_content = table['table_markdown']
                has_number_in_content = f"Tabella: {table_number}" in table_content
                
                print(f"  📋 Tabella {i+1}:")
                print(f"     • Numero globale: {table_number}")
                print(f"     • Pagina: {page}")
                print(f"     • Numero nell'embedding: {'✅' if has_number_in_content else '❌'}")
                
                # Mostra i primi 100 caratteri del contenuto per l'embedding
                preview = table_content[:100].replace('\n', ' ')
                print(f"     • Preview embedding: {preview}...")
                print()
        else:
            print("  ⚪ Nessuna tabella trovata")
            print()
        
        # Test IMMAGINI - verifica numerazione
        print("🔍 ANALISI IMMAGINI:")
        if image_elements:
            for i, image in enumerate(image_elements):
                image_number = image['metadata'].get('image_number', 'N/A')
                page = image['metadata'].get('page', 'N/A')
                
                # Verifica che il numero sia presente nel contenuto per l'embedding
                image_content = image['page_content']
                has_number_in_content = f"Immagine: {image_number}" in image_content
                
                print(f"  🖼️ Immagine {i+1}:")
                print(f"     • Numero globale: {image_number}")
                print(f"     • Pagina: {page}")
                print(f"     • Numero nell'embedding: {'✅' if has_number_in_content else '❌'}")
                
                # Mostra i primi 100 caratteri del contenuto per l'embedding
                preview = image_content[:100].replace('\n', ' ')
                print(f"     • Preview embedding: {preview}...")
                print()
        else:
            print("  ⚪ Nessuna immagine trovata")
            print()
        
        # Verifica sequenzialità della numerazione
        print("🔍 VERIFICA SEQUENZIALITÀ:")
        
        # Tabelle
        table_numbers = [t['metadata'].get('table_number') for t in table_elements if t['metadata'].get('table_number')]
        if table_numbers:
            is_sequential_tables = table_numbers == list(range(1, len(table_numbers) + 1))
            print(f"  📋 Numerazione tabelle sequenziale: {'✅' if is_sequential_tables else '❌'}")
            print(f"     • Numeri trovati: {table_numbers}")
        
        # Immagini
        image_numbers = [i['metadata'].get('image_number') for i in image_elements if i['metadata'].get('image_number')]
        if image_numbers:
            is_sequential_images = image_numbers == list(range(1, len(image_numbers) + 1))
            print(f"  🖼️ Numerazione immagini sequenziale: {'✅' if is_sequential_images else '❌'}")
            print(f"     • Numeri trovati: {image_numbers}")
        
        print()
        print("✅ Test completato!")
        
        # Test di ricerca simulata
        print("\n🔍 SIMULAZIONE QUERY DI RICERCA:")
        print("Esempi di query che ora funzioneranno:")
        if table_numbers:
            for table_num in table_numbers[:3]:  # Prime 3 tabelle
                print(f"  • 'mostra tabella {table_num}' -> Trova la tabella numero {table_num}")
        if image_numbers:
            for image_num in image_numbers[:3]:  # Prime 3 immagini
                print(f"  • 'immagine numero {image_num}' -> Trova l'immagine numero {image_num}")
        
        if not table_numbers and not image_numbers:
            print("  ⚪ Nessun elemento numerato trovato per simulare query")
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_global_numbering()
