"""
Dataset di test di esempio per MultimodalRAG
Questo file contiene esempi di query di test con le relative risposte attese e pagine rilevanti
"""

# Esempio di test queries per documenti scientifici
TEST_QUERIES = [
    {
        "query": "Qual è la definizione di machine learning?",
        "expected_answer": "Il machine learning è un sottocampo dell'intelligenza artificiale che permette ai computer di apprendere e migliorare automaticamente attraverso l'esperienza senza essere esplicitamente programmati.",
        "relevant_pages": ["page_1", "page_2"],
        "query_type": "text",
        "difficulty": "easy"
    },
    {
        "query": "Mostrami il grafico delle performance del modello",
        "expected_answer": "Il grafico mostra che il modello raggiunge un'accuratezza del 95% dopo 100 epoche di training, con una curva di learning che si stabilizza intorno alla 80esima epoca.",
        "relevant_pages": ["figura_3.png", "page_5"],
        "query_type": "multimodal",
        "difficulty": "medium"
    },
    {
        "query": "Quali sono i risultati della tabella di confronto tra i diversi algoritmi?",
        "expected_answer": "La tabella mostra che l'algoritmo Random Forest ottiene i migliori risultati con F1-score di 0.89, seguito da SVM con 0.85 e Neural Network con 0.82.",
        "relevant_pages": ["tabella_2", "page_7"],
        "query_type": "table",
        "difficulty": "medium"
    },
    {
        "query": "Come funziona l'architettura del sistema proposto?",
        "expected_answer": "L'architettura del sistema è composta da tre componenti principali: il modulo di preprocessing che normalizza i dati, il core engine che applica gli algoritmi di ML, e il modulo di output che formatta i risultati.",
        "relevant_pages": ["page_3", "page_4", "diagramma_1.png"],
        "query_type": "multimodal",
        "difficulty": "hard"
    },
    {
        "query": "Quali sono le limitazioni del metodo proposto?",
        "expected_answer": "Le principali limitazioni includono: 1) scalabilità limitata per dataset molto grandi, 2) sensibilità ai parametri di configurazione, 3) necessità di preprocessing intensivo dei dati.",
        "relevant_pages": ["page_8", "page_9"],
        "query_type": "text",
        "difficulty": "hard"
    }
]

# Esempio di valutazione batch
EVALUATION_SCENARIOS = [
    {
        "name": "Basic Text Understanding",
        "description": "Test di comprensione del testo di base",
        "queries": [TEST_QUERIES[0], TEST_QUERIES[4]]
    },
    {
        "name": "Multimodal Content Retrieval",
        "description": "Test di recupero contenuti multimodali",
        "queries": [TEST_QUERIES[1], TEST_QUERIES[3]]
    },
    {
        "name": "Table and Data Analysis",
        "description": "Test di analisi tabelle e dati",
        "queries": [TEST_QUERIES[2]]
    }
]

def get_test_queries():
    """Restituisce la lista delle query di test"""
    return TEST_QUERIES

def get_evaluation_scenarios():
    """Restituisce gli scenari di valutazione"""
    return EVALUATION_SCENARIOS

def get_query_by_difficulty(difficulty: str):
    """Filtra le query per difficoltà"""
    return [q for q in TEST_QUERIES if q["difficulty"] == difficulty]

def get_query_by_type(query_type: str):
    """Filtra le query per tipo"""
    return [q for q in TEST_QUERIES if q["query_type"] == query_type]

# Esempio di utilizzo
if __name__ == "__main__":
    print("📋 Test Dataset per MultimodalRAG")
    print("=" * 50)
    
    print(f"Total queries: {len(TEST_QUERIES)}")
    print(f"Evaluation scenarios: {len(EVALUATION_SCENARIOS)}")
    
    print("\nQueries by difficulty:")
    for difficulty in ["easy", "medium", "hard"]:
        queries = get_query_by_difficulty(difficulty)
        print(f"  {difficulty}: {len(queries)} queries")
    
    print("\nQueries by type:")
    for qtype in ["text", "multimodal", "table"]:
        queries = get_query_by_type(qtype)
        print(f"  {qtype}: {len(queries)} queries")
    
    print("\nExample query:")
    example = TEST_QUERIES[0]
    print(f"Query: {example['query']}")
    print(f"Expected: {example['expected_answer'][:100]}...")
    print(f"Relevant pages: {example['relevant_pages']}")
