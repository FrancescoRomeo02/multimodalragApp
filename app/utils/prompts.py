import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# --- 1. CONFIGURAZIONE ---
# Assicurati che queste costanti siano IDENTICHE a quelle usate nello script di caricamento.
QDRANT_URL = "http://localhost:6333" 
COLLECTION_NAME = "paper_scientifici"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Quanti risultati vogliamo ottenere per ogni domanda?
# 3-5 è un buon numero per la maggior parte dei sistemi RAG.
TOP_K = 3

def main():
    """
    Funzione principale per interrogare la collezione Qdrant.
    """
    print("--- Client di Interrogazione per Paper Scientifici ---")
    print("Sto inizializzando il modello di embedding e il client di Qdrant...")

    # --- 2. INIZIALIZZAZIONE DEI COMPONENTI ---
    # Inizializza il modello di embedding (DEVE essere lo stesso dell'indicizzazione)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Usa 'cuda' se hai una GPU
    )

    # Inizializza il client di Qdrant
    client = QdrantClient(url=QDRANT_URL)

    # Crea un'istanza del vector store di Langchain per interrogare facilmente la collezione
    # Non stiamo aggiungendo documenti, solo collegandoci a quelli esistenti.
    vector_store = Qdrant(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embeddings=embeddings
    )

    print(f"Connesso con successo alla collezione '{COLLECTION_NAME}'.")
    print("Puoi iniziare a fare domande. Digita 'esci' o 'quit' per terminare.")
    print("-" * 50)

    # --- 3. LOOP INTERATTIVO DI DOMANDE E RISPOSTE ---
    while True:
        # Chiedi all'utente di inserire una domanda
        query = input("\nInserisci la tua domanda: ")

        if query.lower() in ['esci', 'quit', 'exit']:
            print("Arrivederci!")
            break

        if not query.strip():
            print("Per favore, inserisci una domanda valida.")
            continue

        print("\nRicerca dei chunk più pertinenti in corso...")

        # Esegui la ricerca di similarità.
        # `similarity_search_with_score` è ottimo perché restituisce anche
        # il punteggio, così puoi vedere QUANTO è pertinente ogni risultato.
        try:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=TOP_K
            )

            if not results:
                print("Nessun risultato trovato per la tua domanda.")
                continue

            # --- 4. VISUALIZZAZIONE DEI RISULTATI ---
            print(f"\n--- Trovati {len(results)} risultati pertinenti ---")
            
            for i, (doc, score) in enumerate(results):
                # Il punteggio di Cosine Similarity è tra 0 e 1. Più alto è, meglio è.
                print(f"\n--- Risultato {i+1} | Punteggio (Similarità): {score:.4f} ---")
                
                # Estrai e stampa i metadati utili
                source = doc.metadata.get('source', 'N/D')
                page = doc.metadata.get('page_number', 'N/D')
                element_type = doc.metadata.get('category', 'N/D') # 'unstructured' usa 'category'

                print(f"Fonte: {os.path.basename(source)} | Pagina: {page} | Tipo Elemento: {element_type}")
                
                # Stampa il contenuto del chunk recuperato
                print("\nContenuto del Chunk:")
                print(">" + "-"*20)
                print(doc.page_content)
                print("<" + "-"*20)

        except Exception as e:
            print(f"Si è verificato un errore durante la ricerca: {e}")
            print("Assicurati che Qdrant sia in esecuzione e che la collezione esista.")

if __name__ == "__main__":
    main()