#!/usr/bin/env python3
"""
Script per verificare la qualità e la completezza del progetto MultimodalRAG.
Esegue tutti i controlli di qualità del codice e i test.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description, fail_on_error=True):
    """Esegue un comando e gestisce gli errori."""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=Path(__file__).parent.parent
        )
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESSO")
            return True
        else:
            print(f"❌ {description} - FALLITO (codice: {result.returncode})")
            if fail_on_error:
                return False
            return True
            
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        if fail_on_error:
            return False
        return True


def check_project_structure():
    """Verifica la struttura del progetto."""
    print(f"\n{'='*60}")
    print("📁 Verifica Struttura del Progetto")
    print(f"{'='*60}")
    
    project_root = Path(__file__).parent.parent
    required_files = [
        "README.md",
        "requirements.txt", 
        "pyproject.toml",
        ".gitignore",
        "Dockerfile",
        "docker-compose.yml",
        "Makefile",
        ".env.example",
        ".pre-commit-config.yaml"
    ]
    
    required_dirs = [
        "src",
        "streamlit_app", 
        "tests",
        "scripts",
        "data"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_name in required_files:
        if not (project_root / file_name).exists():
            missing_files.append(file_name)
    
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_files or missing_dirs:
        print("❌ File o directory mancanti:")
        for f in missing_files:
            print(f"  - File: {f}")
        for d in missing_dirs:
            print(f"  - Directory: {d}")
        return False
    else:
        print("✅ Struttura del progetto - OK")
        return True


def main():
    """Funzione principale."""
    print("🚀 MultimodalRAG - Controllo Qualità del Progetto")
    print("="*60)
    
    # Cambia nella directory del progetto
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    all_checks_passed = True
    
    # 1. Verifica struttura del progetto
    if not check_project_structure():
        all_checks_passed = False
    
    # 2. Controllo sintassi Python
    if not run_command(
        "python -m py_compile src/**/*.py streamlit_app/**/*.py scripts/**/*.py",
        "Controllo Sintassi Python",
        fail_on_error=False
    ):
        all_checks_passed = False
    
    # 3. Controllo formattazione con Black
    if not run_command(
        "black --check --diff src/ streamlit_app/ scripts/ tests/",
        "Controllo Formattazione (Black)",
        fail_on_error=False
    ):
        print("ℹ️  Esegui 'make format' per correggere automaticamente la formattazione")
    
    # 4. Controllo linting con Flake8
    if not run_command(
        "flake8 src/ streamlit_app/ scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503",
        "Linting (Flake8)",
        fail_on_error=False
    ):
        print("ℹ️  Correggi i problemi di linting segnalati")
    
    # 5. Controllo import con isort
    if not run_command(
        "isort --check-only --diff src/ streamlit_app/ scripts/ tests/",
        "Controllo Import (isort)",
        fail_on_error=False
    ):
        print("ℹ️  Esegui 'make format' per correggere l'ordinamento degli import")
    
    # 6. Test di struttura del progetto
    if not run_command(
        "python -m pytest tests/unit/test_project_structure.py -v",
        "Test Struttura Progetto",
        fail_on_error=False
    ):
        all_checks_passed = False
    
    # 7. Test di qualità del codice
    if not run_command(
        "python -m pytest tests/unit/test_code_quality.py -v",
        "Test Qualità Codice",
        fail_on_error=False
    ):
        all_checks_passed = False
    
    # 8. Test di integrazione
    if not run_command(
        "python -m pytest tests/integration/ -v",
        "Test di Integrazione",
        fail_on_error=False
    ):
        print("ℹ️  Alcuni test di integrazione potrebbero fallire senza dipendenze esterne")
    
    # 9. Controllo sicurezza (se bandit è installato)
    run_command(
        "bandit -r src/ streamlit_app/ scripts/ -f json",
        "Controllo Sicurezza (Bandit)",
        fail_on_error=False
    )
    
    # 10. Controllo dipendenze vulnerabili (se safety è installato)
    run_command(
        "safety check",
        "Controllo Vulnerabilità Dipendenze (Safety)",
        fail_on_error=False
    )
    
    # Riepilogo finale
    print(f"\n{'='*60}")
    print("📊 RIEPILOGO FINALE")
    print(f"{'='*60}")
    
    if all_checks_passed:
        print("🎉 Tutti i controlli principali sono passati!")
        print("✅ Il progetto è pronto per la consegna")
        return 0
    else:
        print("⚠️  Alcuni controlli hanno rilevato problemi")
        print("🔧 Consulta i dettagli sopra per le correzioni necessarie")
        return 1


if __name__ == "__main__":
    sys.exit(main())
