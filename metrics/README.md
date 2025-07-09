# Sistema di Metriche MultimodalRAG

Questo documento descrive il nuovo sistema di metriche implementato per MultimodalRAG, che sostituisce il precedente sistema di monitoring basato su Prometheus/Grafana.

## Panoramica

Il sistema di metriche traccia e analizza le performance del sistema RAG attraverso quattro categorie principali:

1. **Retrieval Metrics** - Metriche di recupero documenti
2. **Generation Metrics** - Metriche di generazione del testo
3. **Multimodal Metrics** - Metriche specifiche per contenuti multimodali
4. **Cost & Efficiency Metrics** - Metriche di costo e efficienza

## Tipologie di Metriche

### 1. Retrieval Metrics

Valutano l'efficacia del sistema di recupero documenti:

- **Recall@k**: Frazione di documenti rilevanti recuperati nei primi k risultati
- **Precision@k**: Frazione di documenti rilevanti tra i primi k risultati recuperati
- **Mean Reciprocal Rank (MRR)**: Posizione media del primo documento rilevante


### 2. Generation Metrics

Valutano la qualità del testo generato:

- **BERTScore**: Confronto semantico usando embeddings BERT (Precision, Recall, F1)
- **Exact Match (EM)**: Match esatto tra testo generato e riferimento


### 3. Multimodal Metrics

Valutano l'accuratezza del recupero di contenuti multimodali:

- **Image/Text Accuracy**: Accuratezza combinata di recupero immagini e testo


### 4. Cost & Efficiency Metrics

Monitorano costi computazionali ed efficienza:

- **Cost (token number)**: Numero totale di token utilizzati
- **Time per response**: Tempo di risposta totale
- **Embedding time and cost**: Tempo e token per gli embedding


## Utilizzo del Collettore di Metriche

### Accesso al Collettore Globale

```python
from metrics.main_metrics import global_metrics_collector

# Ottieni summary delle metriche di retrieval
retrieval_summary = global_metrics_collector.get_retrieval_summary()

# Ottieni summary completo di tutte le metriche
complete_summary = global_metrics_collector.get_complete_summary()

# Esporta metriche in JSON
export_path = global_metrics_collector.export_metrics("my_metrics.json")

# Salva sessione corrente
session_path = global_metrics_collector.save_session("experiment_1")
```

### Summary delle Metriche

Ogni categoria di metriche ha il suo summary specifico:

```python
# Retrieval summary
retrieval_summary = global_metrics_collector.get_retrieval_summary()
print(f"Average MRR: {retrieval_summary['mean_reciprocal_rank']}")

# Generation summary  
generation_summary = global_metrics_collector.get_generation_summary()
print(f"Average BERT F1: {generation_summary['bert_score']['avg_f1']}")

# Multimodal summary
multimodal_summary = global_metrics_collector.get_multimodal_summary()
print(f"Image/Text accuracy: {multimodal_summary['avg_image_text_accuracy']}")

# Cost & efficiency summary
cost_summary = global_metrics_collector.get_cost_efficiency_summary()
print(f"Average tokens per query: {cost_summary['avg_tokens_per_query']}")
```

## Struttura dei File Esportati

I file JSON esportati contengono:

```json
{
  "export_timestamp": "2025-01-09T...",
  "summary": {
    "retrieval_metrics": {...},
    "generation_metrics": {...},
    "multimodal_metrics": {...},
    "cost_efficiency_metrics": {...}
  },
  "raw_data": {
    "retrieval": [...],
    "generation": [...],
    "multimodal": [...],
    "cost_efficiency": [...]
  }
}
```

## Dipendenze

Il sistema richiede le seguenti dipendenze (già incluse in requirements.txt):

- `bert_score>=0.3.13` - Per le metriche BERTScore
- `nltk>=3.8.0` - Per elaborazione testo
- `numpy` - Per calcoli numerici
- `pandas` - Per manipolazione dati

## Note

- Le metriche vengono memorizzate in memoria durante l'esecuzione
- Utilizzare `export_metrics()` per persistere i dati
- Il sistema è progettato per essere leggero e non interferire con le performance
- BERTScore è opzionale - se non disponibile, restituisce valori 0.0

