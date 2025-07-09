#!/usr/bin/env python3
"""
Test script per il nuovo parser PDF unificato.
Testa sia la modalità VLM che quella di fallback.
"""

import sys
import os
import time
from pathlib import Path

# Aggiungi il percorso del progetto
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.pdf_parser_unified import parse_pdf_elements_unified, parse_pdf_elements
from src.config import RAW_DATA_PATH

def test_unified_parser():
    """Test del parser unificato."""
    
    # Trova un PDF di test
    test_files = []
    if os.path.exists(RAW_DATA_PATH):
        test_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.pdf')]
    
    if not test_files:
        print("❌ Nessun file PDF trovato in data/raw/")
        print("💡 Copia un file PDF in data/raw/ per testare il parser")
        return
    
    test_file = os.path.join(RAW_DATA_PATH, test_files[0])
    print(f"🧪 Test del parser unificato con: {test_files[0]}")
    print("=" * 60)
    
    # Test 1: Parser standard (compatibilità)
    print("\n1️⃣ Test parser standard (compatibilità):")
    start_time = time.time()
    try:
        texts, images, tables = parse_pdf_elements(test_file)
        elapsed = time.time() - start_time
        
        print(f"✅ Successo in {elapsed:.2f}s")
        print(f"   📝 Testi: {len(texts)}")
        print(f"   🖼️  Immagini: {len(images)}")
        print(f"   📊 Tabelle: {len(tables)}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
    
    # Test 2: Parser unificato senza VLM
    print("\n2️⃣ Test parser unificato (senza VLM):")
    start_time = time.time()
    try:
        texts, images, tables, stats = parse_pdf_elements_unified(test_file, use_vlm=False)
        elapsed = time.time() - start_time
        
        print(f"✅ Successo in {elapsed:.2f}s")
        print(f"   📝 Testi: {len(texts)}")
        print(f"   🖼️  Immagini: {len(images)}")
        print(f"   📊 Tabelle: {len(tables)}")
        print(f"   📑 Pagine: {stats.total_pages}")
        print(f"   ⏱️  Tempo parsing: {stats.total_time:.2f}s")
        print(f"   🧠 VLM usato: {'✓' if stats.vlm_analysis_used else '✗'}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
    
    # Test 3: Parser unificato con VLM
    print("\n3️⃣ Test parser unificato (con VLM):")
    start_time = time.time()
    try:
        texts, images, tables, stats = parse_pdf_elements_unified(test_file, use_vlm=True)
        elapsed = time.time() - start_time
        
        print(f"✅ Successo in {elapsed:.2f}s")
        print(f"   📝 Testi: {len(texts)}")
        print(f"   🖼️  Immagini: {len(images)}")
        print(f"   📊 Tabelle: {len(tables)}")
        print(f"   📑 Pagine: {stats.total_pages}")
        print(f"   ⏱️  Tempo parsing: {stats.total_time:.2f}s")
        print(f"   🧠 VLM usato: {'✓' if stats.vlm_analysis_used else '✗'}")
        if stats.vlm_analysis_used:
            print(f"   🤖 Pagine VLM: {stats.vlm_pages_analyzed}")
            print(f"   🔄 Pagine fallback: {stats.fallback_pages}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Test completato!")

if __name__ == "__main__":
    test_unified_parser()
