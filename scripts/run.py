#!/usr/bin/env python3
"""
Script di avvio per MultimodalRAG.

Questo script fornisce diversi modi per avviare l'applicazione:
- Modalità locale con Streamlit
- Modalità Docker
- Modalità sviluppo con hot-reload
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Aggiungi il root del progetto al Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def run_streamlit():
    """Avvia l'applicazione Streamlit in modalità locale."""
    cmd = [
        "streamlit", "run", 
        str(ROOT_DIR / "streamlit_app" / "Home.py"),
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    print(f"🚀 Avvio Streamlit: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR)


def run_docker_build():
    """Costruisce l'immagine Docker."""
    cmd = ["docker", "build", "-t", "multimodalrag", "."]
    print(f"Build Docker: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def run_docker():
    """Avvia l'applicazione in modalità Docker."""
    # Prima costruisci l'immagine
    run_docker_build()
    
    # Poi esegui il container
    cmd = [
        "docker", "run", "-d", 
        "-p", "8501:8501", 
        "--name", "multimodalrag_container", 
        "multimodalrag"
    ]
    print(f"Avvio Docker: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def run_docker_compose():
    """Avvia l'applicazione con Docker Compose."""
    cmd = ["docker-compose", "up", "-d"]
    print(f"Avvio Docker Compose: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def run_dev():
    """Avvia in modalità sviluppo con hot-reload."""
    cmd = [
        "streamlit", "run", 
        str(ROOT_DIR / "streamlit_app" / "Home.py"),
        "--server.port=8501",
        "--server.address=localhost",
        "--server.runOnSave=true",
        "--server.fileWatcherType=auto"
    ]
    print(f"Avvio modalità sviluppo: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Avvia MultimodalRAG")
    parser.add_argument(
        "mode",
        choices=["local", "docker", "docker-compose", "dev", "build"],
        help="Modalità di avvio"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Avvio MultimodalRAG in modalità: {args.mode}")
    print(f"📁 Directory di lavoro: {ROOT_DIR}")
    
    try:
        if args.mode == "local":
            run_streamlit()
        elif args.mode == "docker":
            run_docker()
        elif args.mode == "docker-compose":
            run_docker_compose()
        elif args.mode == "dev":
            run_dev()
        elif args.mode == "build":
            run_docker_build()
    except KeyboardInterrupt:
        print("\n⏹️  Arresto dell'applicazione...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()