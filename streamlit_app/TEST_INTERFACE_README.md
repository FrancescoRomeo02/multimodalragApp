# Test & Evaluation Interface

Interfaccia dedicata per testare e valutare automaticamente le performance del sistema MultimodalRAG.

## Avvio

```bash
# Tramite Makefile
make run-test-interface

# O direttamente
python scripts/run_test_interface.py
```

L'interfaccia sarà disponibile su [http://localhost:8503](http://localhost:8503)

## Funzionalità

### Input di Test
- **Query**: La domanda da testare
- **Risposta Attesa**: La risposta corretta per valutare la generazione
- **Pagine Rilevanti**: I documenti/pagine che contengono la risposta (per valutare il retrieval)

### Metriche Automatiche
- **Retrieval**: Recall@k, Precision@k, MRR
- **Generation**: BERTScore, Exact Match
- **Multimodal**: Image/Text accuracy
- **Cost & Efficiency**: Tokens, timing, throughput

### Visualizzazione
- Confronto side-by-side tra risposta attesa e generata
- Analisi delle pagine recuperate vs rilevanti
- Metriche dettagliate con spiegazioni
- Export dei risultati in JSON

## Esempio di Utilizzo

1. **Carica documenti** tramite la sidebar
2. **Inserisci query**: "Qual è la definizione di machine learning?"
3. **Inserisci risposta attesa**: "Il machine learning è..."
4. **Inserisci pagine rilevanti**: 
   ```
   page_1
   page_2
   introduction.pdf
   ```
5. **Clicca "Esegui Test"** per ottenere tutte le metriche automaticamente

## Dataset di Test

Usa il dataset di esempio in `test_data/sample_test_dataset.py`:

```python
from test_data.sample_test_dataset import get_test_queries

queries = get_test_queries()
for query in queries:
    print(f"Query: {query['query']}")
    print(f"Expected: {query['expected_answer']}")
    print(f"Relevant: {query['relevant_pages']}")
```

## Casi d'Uso

- **Valutazione di modelli**: Confronta performance di diversi LLM
- **Benchmark**: Testa su dataset standardizzati
- **Debug**: Identifica errori di retrieval e generazione
- **Ottimizzazione**: Misura l'impatto di modifiche al sistema
- **Reporting**: Genera report dettagliati per ricerca

## Output

I risultati vengono salvati in:
- `metrics_data/test_session_metrics.json` - Metriche complete
- Session state per visualizzazione immediata
- Export personalizzati tramite l'interfaccia

## Personalizzazione

L'interfaccia può essere estesa per:
- Carattamentali di test personalizzati
- Metriche domain-specific
- Integrazione con dataset esterni
- Valutazione batch automatica
