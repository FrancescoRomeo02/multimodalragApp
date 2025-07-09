#!/usr/bin/env python3
"""
Script per avviare l'interfaccia di test e valutazione di MultimodalRAG
"""

import sys
import os
import subprocess

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def main():
    """Avvia l'interfaccia di test"""
    print("🧪 Starting MultimodalRAG Test & Evaluation Interface...")
    print("=" * 60)
    print("This interface allows you to:")
    print("- Input test queries with expected answers")
    print("- Specify relevant pages/documents for evaluation")
    print("- Automatically calculate all RAG metrics")
    print("- View detailed performance analysis")
    print("=" * 60)
    
    # Path to the test evaluation app
    test_app_path = os.path.join(PROJECT_ROOT, "streamlit_app", "test_evaluation.py")
    
    # Run streamlit
    try:
        subprocess.run([
            "streamlit", "run", test_app_path,
            "--server.port=8503",  # Different port from main app
            "--server.address=0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Test evaluation interface stopped.")

if __name__ == "__main__":
    main()
