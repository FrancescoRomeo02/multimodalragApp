#!/usr/bin/env python3
"""
Test finale dopo la pulizia del codice.
Verifica che tutte le funzioni principali funzionino correttamente.
"""

import sys
import os

# Aggiungi il percorso del progetto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def test_imports():
    """Testa che tutti i moduli principali possano essere importati"""
    print("🔍 Testing imports...")
    
    try:
        from src.pipeline.retriever import enhanced_rag_query
        print("✅ Retriever import OK")
        
        from src.utils.embedder import get_embedding_model
        print("✅ Embedder import OK")
        
        from src.pipeline.indexer_service import DocumentIndexer
        print("✅ Indexer import OK")
        
        from src.utils.qdrant_utils import qdrant_manager
        print("✅ Qdrant utils import OK")
        
        from streamlit_app.components.ui_components import enhanced_chat_interface_widget
        print("✅ UI components import OK")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_embedder():
    """Testa il funzionamento dell'embedder"""
    print("\n🔍 Testing embedder...")
    
    try:
        from src.utils.embedder import get_embedding_model
        embedder = get_embedding_model()
        
        # Test embedding di una frase
        test_text = "Questo è un testo di prova per il sistema RAG multimodale"
        embedding = embedder.embed_query(test_text)
        
        if len(embedding) > 0:
            print(f"✅ Embedder working, embedding dimension: {len(embedding)}")
            return True
        else:
            print("❌ Embedder returned empty embedding")
            return False
            
    except Exception as e:
        print(f"❌ Embedder error: {e}")
        return False

def test_retriever_function():
    """Testa la funzione enhanced_rag_query (senza necessariamente database)"""
    print("\n🔍 Testing retriever function...")
    
    try:
        from src.pipeline.retriever import enhanced_rag_query
        
        # Test dei parametri della funzione
        import inspect
        sig = inspect.signature(enhanced_rag_query)
        params = list(sig.parameters.keys())
        
        expected_params = ['query', 'selected_files']
        if params == expected_params:
            print(f"✅ Function signature correct: {params}")
            return True
        else:
            print(f"❌ Function signature mismatch. Expected: {expected_params}, Got: {params}")
            return False
            
    except Exception as e:
        print(f"❌ Retriever function error: {e}")
        return False

def test_config():
    """Testa la configurazione"""
    print("\n🔍 Testing configuration...")
    
    try:
        from src.config import validate_config, DEFAULT_EMBEDDING_MODEL, COLLECTION_NAME
        
        # Verifica configurazione
        config_error = validate_config()
        if config_error:
            print(f"⚠️ Config issue: {config_error}")
        else:
            print("✅ Configuration valid")
        
        print(f"✅ Using embedding model: {DEFAULT_EMBEDDING_MODEL}")
        print(f"✅ Using collection: {COLLECTION_NAME}")
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_no_legacy_functions():
    """Verifica che le funzioni legacy siano state rimosse"""
    print("\n🔍 Testing that legacy functions are removed...")
    
    try:
        from src.pipeline import retriever
        
        # Verifica che le funzioni legacy non esistano più
        legacy_functions = ['create_retriever', 'create_rag_chain', 'batch_query', 'edit_answer']
        
        for func_name in legacy_functions:
            if hasattr(retriever, func_name):
                print(f"❌ Legacy function still exists: {func_name}")
                return False
        
        print("✅ All legacy functions have been removed")
        return True
        
    except Exception as e:
        print(f"❌ Legacy function check error: {e}")
        return False

def main():
    """Esegue tutti i test"""
    print("🧹 CLEAN CODE TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_embedder, 
        test_retriever_function,
        test_config,
        test_no_legacy_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Code cleanup successful!")
        return True
    else:
        print("⚠️ Some tests failed. Review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
