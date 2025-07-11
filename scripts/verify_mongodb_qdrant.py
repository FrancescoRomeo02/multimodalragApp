#!/usr/bin/env python3
"""
Script di diagnostica per verificare la sincronizzazione tra Qdrant e MongoDB.
Questo script analizza i dati presenti in entrambi i database per verificare che:
1. Tutti i documenti in Qdrant abbiano un riferimento valido a MongoDB (mongo_id)
2. Tutti i tipi di contenuto (testo, immagini, tabelle) siano correttamente salvati in MongoDB
3. Non ci siano riferimenti orfani in entrambe le direzioni

Utilizzo:
    python scripts/verify_mongodb_qdrant.py
"""

import os
import sys
import logging
from collections import Counter
import argparse
from typing import Dict, List, Tuple, Set
from tabulate import tabulate

# Aggiungi la directory parent alla path per l'import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.qdrant_utils import qdrant_manager
from src.utils.mongodb_utils import mongodb_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_qdrant_stats() -> Dict:
    """
    Recupera statistiche dai documenti in Qdrant.
    
    Returns:
        Dizionario con statistiche sui documenti in Qdrant
    """
    try:
        # Ottieni tutte le collection in Qdrant
        collections_info = qdrant_manager.client.get_collections()
        collections = [c.name for c in collections_info.collections]
        
        if not collections:
            logger.warning("Nessuna collection trovata in Qdrant!")
            return {"collections": [], "total_points": 0}
        
        total_points = 0
        collection_stats = {}
        
        for coll in collections:
            try:
                # Ottieni conteggio dei punti nella collection
                count = qdrant_manager.client.get_collection(coll).vectors_count
                count_value = count if count is not None else 0
                collection_stats[coll] = count_value
                total_points += count_value
            except Exception as e:
                logger.error(f"Errore recuperando statistiche per collection {coll}: {e}")
        
        return {
            "collections": collections,
            "collection_stats": collection_stats,
            "total_points": total_points
        }
    except Exception as e:
        logger.error(f"Errore recuperando statistiche da Qdrant: {e}")
        return {"error": str(e)}

def get_qdrant_mongodb_references() -> Tuple[Dict, List]:
    """
    Analizza i riferimenti tra Qdrant e MongoDB.
    
    Returns:
        Tuple con statistiche sui riferimenti e lista di documenti con problemi
    """
    try:
        # Recupera i riferimenti da Qdrant a MongoDB
        qdrant_to_mongodb_refs = {}
        qdrant_sources = set()
        problem_documents = []
        
        # Prendi solo la collection principale
        collection_name = qdrant_manager.collection_name
        
        # Recupera tutti i documenti dalla collection principale
        limit = 10000  # Limita il numero di documenti per evitare problemi di memoria
        offset = 0
        batch_size = 100
        
        while True:
            documents_result = qdrant_manager.client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            documents, next_offset = documents_result
            
            if not documents:
                break
                
            for point in documents:
                point_id = point.id
                payload = point.payload or {}
                metadata = payload.get("metadata", {})
                source = metadata.get("source", "unknown")
                qdrant_sources.add(source)
                
                # Verifica se c'è un riferimento a MongoDB
                mongo_id = metadata.get("mongo_id")
                content_type = metadata.get("origin_type", "unknown")
                
                if not mongo_id:
                    problem_documents.append({
                        "database": "Qdrant",
                        "id": point_id,
                        "source": source,
                        "problem": "Missing MongoDB reference",
                        "content_type": content_type
                    })
                    continue
                    
                # Verifica se il riferimento è valido in MongoDB
                mongo_doc = mongodb_manager.get_document_by_id(mongo_id)
                if not mongo_doc:
                    problem_documents.append({
                        "database": "Qdrant",
                        "id": point_id,
                        "mongo_id": mongo_id,
                        "source": source,
                        "problem": "Invalid MongoDB reference",
                        "content_type": content_type
                    })
                    continue
                    
                # Aggiorna statistiche
                qdrant_to_mongodb_refs[mongo_id] = {
                    "qdrant_id": point_id,
                    "source": source,
                    "content_type": content_type
                }
            
            offset += batch_size
            if offset >= limit:
                break
        
        # Controlla documenti orfani in MongoDB
        mongo_sources = set()
        mongodb_orphans = []
        
        # Recupera tutti i documenti da MongoDB
        mongodb_docs = list(mongodb_manager.collection.find({}, {"_id": 1, "source": 1, "content_type": 1}))
        
        for doc in mongodb_docs:
            mongo_id = str(doc["_id"])
            source = doc.get("source", "unknown")
            content_type = doc.get("content_type", "unknown")
            mongo_sources.add(source)
            
            # Verifica se il documento è riferito da Qdrant
            if mongo_id not in qdrant_to_mongodb_refs:
                mongodb_orphans.append({
                    "database": "MongoDB",
                    "id": mongo_id,
                    "source": source,
                    "problem": "Not referenced by Qdrant",
                    "content_type": content_type
                })
        
        # Calcola statistiche sui riferimenti
        total_qdrant_docs = qdrant_manager.client.get_collection(collection_name).vectors_count
        total_mongodb_docs = mongodb_manager.collection.count_documents({})
        
        # Statistiche per tipo di contenuto in MongoDB
        content_types_mongodb = Counter()
        for doc_type in ["text", "image", "table"]:
            count = mongodb_manager.collection.count_documents({"content_type": doc_type})
            content_types_mongodb[doc_type] = count
        
        stats = {
            "total_qdrant_documents": total_qdrant_docs,
            "total_mongodb_documents": total_mongodb_docs,
            "valid_references": len(qdrant_to_mongodb_refs),
            "qdrant_missing_references": len([d for d in problem_documents if d["database"] == "Qdrant" and d["problem"] == "Missing MongoDB reference"]),
            "qdrant_invalid_references": len([d for d in problem_documents if d["database"] == "Qdrant" and d["problem"] == "Invalid MongoDB reference"]),
            "mongodb_orphans": len(mongodb_orphans),
            "qdrant_sources": list(qdrant_sources),
            "mongodb_sources": list(mongo_sources),
            "content_types_mongodb": dict(content_types_mongodb)
        }
        
        # Unisci i documenti problematici
        all_problems = problem_documents + mongodb_orphans
        
        return stats, all_problems
    
    except Exception as e:
        logger.error(f"Errore analizzando riferimenti tra Qdrant e MongoDB: {e}")
        return {"error": str(e)}, []

def main():
    """Funzione principale che esegue la verifica."""
    parser = argparse.ArgumentParser(description="Verifica la sincronizzazione tra Qdrant e MongoDB")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mostra informazioni dettagliate")
    args = parser.parse_args()
    
    logger.info("Iniziando la verifica della sincronizzazione tra Qdrant e MongoDB...")
    
    # Verifica connessione ai database
    qdrant_connected = qdrant_manager.verify_connection()
    mongodb_connected = mongodb_manager.verify_connection()
    
    if not qdrant_connected:
        logger.error("Impossibile connettersi a Qdrant!")
        return
        
    if not mongodb_connected:
        logger.error("Impossibile connettersi a MongoDB!")
        return
    
    # Ottieni statistiche di Qdrant
    qdrant_stats = get_qdrant_stats()
    logger.info(f"Statistiche Qdrant: {qdrant_stats}")
    
    # Ottieni riferimenti tra Qdrant e MongoDB
    ref_stats, problem_docs = get_qdrant_mongodb_references()
    
    # Stampa risultati
    print("\n==== RISULTATI VERIFICA SINCRONIZZAZIONE QDRANT-MONGODB ====")
    
    print("\nSTATISTICHE GENERALI:")
    general_stats = [
        ["Totale documenti Qdrant", ref_stats.get("total_qdrant_documents", "N/A")],
        ["Totale documenti MongoDB", ref_stats.get("total_mongodb_documents", "N/A")],
        ["Riferimenti validi", ref_stats.get("valid_references", "N/A")],
        ["Documenti Qdrant senza riferimento MongoDB", ref_stats.get("qdrant_missing_references", "N/A")],
        ["Documenti Qdrant con riferimento MongoDB invalido", ref_stats.get("qdrant_invalid_references", "N/A")],
        ["Documenti MongoDB orfani", ref_stats.get("mongodb_orphans", "N/A")]
    ]
    print(tabulate(general_stats, tablefmt="grid"))
    
    print("\nTIPOLOGIE DI CONTENUTO IN MONGODB:")
    content_types = ref_stats.get("content_types_mongodb", {})
    content_stats = [[k, v] for k, v in content_types.items()]
    print(tabulate(content_stats, headers=["Tipo", "Conteggio"], tablefmt="grid"))
    
    # Mostra fonti se verbose
    if args.verbose:
        print("\nFONTI PRESENTI:")
        sources_data = []
        qdrant_sources = set(ref_stats.get("qdrant_sources", []))
        mongo_sources = set(ref_stats.get("mongodb_sources", []))
        all_sources = sorted(qdrant_sources.union(mongo_sources))
        
        for source in all_sources:
            sources_data.append([
                source,
                "✓" if source in qdrant_sources else "✗",
                "✓" if source in mongo_sources else "✗"
            ])
        
        print(tabulate(sources_data, headers=["Fonte", "In Qdrant", "In MongoDB"], tablefmt="grid"))
    
    # Mostra problemi
    if problem_docs:
        print(f"\nPROBLEMI TROVATI ({len(problem_docs)}):")
        if args.verbose:
            # Mostra tutti i problemi in modalità verbosa
            problem_data = [[d.get("database"), d.get("id"), d.get("source"), d.get("content_type", "N/A"), d.get("problem")] for d in problem_docs[:20]]
            print(tabulate(problem_data, headers=["Database", "ID", "Fonte", "Tipo contenuto", "Problema"], tablefmt="grid"))
            
            if len(problem_docs) > 20:
                print(f"... e altri {len(problem_docs) - 20} problemi.")
        else:
            # Mostra solo statistiche in modalità non verbosa
            problem_types = Counter(f"{d.get('database')} - {d.get('problem')}" for d in problem_docs)
            problem_stats = [[problem, count] for problem, count in problem_types.items()]
            print(tabulate(problem_stats, headers=["Problema", "Conteggio"], tablefmt="grid"))
    else:
        print("\nNessun problema trovato! I database sono correttamente sincronizzati.")
    
    print("\n==== CONCLUSIONE ====")
    if ref_stats.get("qdrant_missing_references", 0) > 0 or ref_stats.get("qdrant_invalid_references", 0) > 0 or ref_stats.get("mongodb_orphans", 0) > 0:
        print("⚠️  ATTENZIONE: Sono stati trovati problemi di sincronizzazione tra Qdrant e MongoDB.")
        print("   Si consiglia di ricostruire l'indice o correggere manualmente i riferimenti.")
    else:
        print("✅ I database Qdrant e MongoDB sono correttamente sincronizzati!")
        
    # Verifica se tutti i tipi di contenuto sono presenti in MongoDB
    if all(content_types.get(t, 0) > 0 for t in ["text", "image", "table"]):
        print("✅ Tutti i tipi di contenuto (testo, immagini, tabelle) sono presenti in MongoDB.")
    else:
        missing = [t for t in ["text", "image", "table"] if content_types.get(t, 0) == 0]
        print(f"⚠️  ATTENZIONE: Alcuni tipi di contenuto non sono presenti in MongoDB: {', '.join(missing)}.")
        print("   Assicurarsi che tutti i tipi di contenuto vengano elaborati e salvati correttamente.")

if __name__ == "__main__":
    main()
