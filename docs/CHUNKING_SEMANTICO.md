# Chunking Semantico - Guida Completa

## Panoramica

Il sistema di **chunking semantico** è stato integrato nel multimodal RAG per migliorare la qualità della segmentazione dei documenti. A differenza del chunking classico che divide il testo in base alla lunghezza, il chunking semantico analizza il contenuto e crea segmenti basati sulla coerenza semantica.

## Caratteristiche Principali

### Chunking Semantico Avanzato
- **Analisi semantica**: Utilizza embeddings per comprendere il significato del testo
- **Soglie configurabili**: Personalizzazione dei parametri di similarità
- **Fallback intelligente**: Ricade sul chunking classico in caso di errori
- **Preservazione metadati**: Mantiene tutte le informazioni contestuali

### Gestione Multimodale
- **Associazione intelligente**: Collega immagini e tabelle ai chunk di testo rilevanti
- **Metadati arricchiti**: Aggiunge informazioni sulle relazioni tra elementi
- **Contesto preservato**: Mantiene le descrizioni AI e il contesto manuale

### Monitoraggio Performance
- **Metriche dettagliate**: Traccia performance e qualità del chunking
- **Confronti automatici**: Valuta semantico vs classico
- **Grafici di analisi**: Visualizzazioni delle performance
- **Export dati**: Esportazione in CSV per analisi esterne

## Configurazione

### Parametri Chunking Semantico

Aggiunti in `src/config.py`:

```python
# SEMANTIC CHUNKING SETTINGS
SEMANTIC_CHUNK_SIZE = 1000              # Dimensione target dei chunk
SEMANTIC_CHUNK_OVERLAP = 200            # Sovrapposizione tra chunk
SEMANTIC_THRESHOLD = 0.75               # Soglia similarità semantica
MIN_CHUNK_SIZE = 100                    # Dimensione minima chunk
CHUNKING_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
```

### Nuove Dipendenze

```bash
pip install langchain-text-splitters langchain-experimental tiktoken
pip install matplotlib seaborn pandas  # Per monitoraggio
```

## Utilizzo


```python
from src.utils.semantic_chunker import MultimodalSemanticChunker
from src.config import *

# Inizializzazione
chunker = MultimodalSemanticChunker(
    embedding_model=CHUNKING_EMBEDDING_MODEL,
    chunk_size=SEMANTIC_CHUNK_SIZE,
    chunk_overlap=SEMANTIC_CHUNK_OVERLAP,
    semantic_threshold=SEMANTIC_THRESHOLD,
    min_chunk_size=MIN_CHUNK_SIZE
)

# Chunking testo
text_elements = [...] # I tuoi elementi testuali
chunks = chunker.chunk_text_elements(text_elements)

# Associazione multimediale
enriched_texts, enriched_images, enriched_tables = \
    chunker.associate_media_to_chunks(chunks, images, tables)

# Statistiche
stats = chunker.get_chunking_stats(enriched_texts)
```


## Integrazione con Indexer

### DocumentIndexer Aggiornato

```python
from src.pipeline.indexer_service import DocumentIndexer

# Con chunking semantico (default)
indexer = DocumentIndexer(embedder, use_semantic_chunking=True)

# Con chunking classico
indexer = DocumentIndexer(embedder, use_semantic_chunking=False)
```

### Metadati Arricchiti

I chunk semantici includono:

```python
{
    "text": "...",
    "metadata": {
        "chunk_id": "1_0",               # ID univoco chunk
        "chunk_index": 0,                # Indice nel documento
        "total_chunks": 5,               # Totale chunk documento
        "chunk_type": "semantic",        # Tipo chunking
        "original_length": 2000,         # Lunghezza testo originale
        "chunk_length": 400,             # Lunghezza chunk
        "related_images": 2,             # Immagini correlate
        "related_tables": 1,             # Tabelle correlate
        # ... altri metadati originali
    }
}
```

## Vantaggi del Chunking Semantico

### Qualità Migliorata
- **Coerenza semantica**: Chunk più significativi
- **Contesto preservato**: Mantiene il senso delle frasi
- **Soglie adattive**: Si adatta al contenuto

### Ricerca Più Efficace
- **Rilevanza aumentata**: Chunk semanticamente coerenti
- **Contesto migliore**: Relazioni preserve tra concetti
- **Precisione superiore**: Risultati più pertinenti

### Flessibilità
- **Configurabile**: Parametri personalizzabili
- **Fallback sicuro**: Gestione errori robusta
- **Multimodale**: Integrazione con immagini e tabelle

## Troubleshooting

### Problemi Comuni

1. **Chunking troppo lento**
   - Riduci `SEMANTIC_CHUNK_SIZE`
   - Usa modello embedding più leggero
   - Aumenta `SEMANTIC_THRESHOLD`

2. **Troppi chunk piccoli**
   - Aumenta `MIN_CHUNK_SIZE`
   - Riduci `SEMANTIC_THRESHOLD`
   - Verifica qualità del testo input

3. **Fallback frequenti**
   - Controlla connessione modello embedding
   - Verifica formato testo input
   - Aumenta timeout/memoria

### Log e Debug

```python
import logging
logging.getLogger('src.utils.semantic_chunker').setLevel(logging.DEBUG)
```

## Conclusioni

Il chunking semantico rappresenta un significativo miglioramento per la qualità del RAG multimodale, offrendo:

- Segmentazione più intelligente e significativa
- Migliore preservazione del contesto
- Integrazione ottimizzata con elementi multimediali
- Monitoraggio completo delle performance

Raccomandiamo l'uso del chunking semantico per la maggior parte dei casi d'uso, mantenendo il chunking classico come fallback sicuro.
