# Numerazione Globale di Immagini e Tabelle negli Embedding

## 🎯 Problema Risolto

**PRIMA**: Il contenuto degli embedding per immagini e tabelle non includeva informazioni sulla numerazione, rendendo impossibile query specifiche come "mostra tabella 3" o "immagine numero 2".

**DOPO**: Ogni immagine e tabella ha ora:
- Numerazione globale sequenziale nel documento
- Numero incluso nel contenuto dell'embedding
- Numero disponibile nei metadati per query dirette

## ✅ Implementazione

### 📊 Tabelle

```python
# Ogni tabella ora include:
{
  "table_markdown": "Tabella: 1\n[contenuto markdown della tabella]",
  "table_markdown_raw": "[contenuto originale senza numerazione]",
  "metadata": {
    "table_number": 1,
    "type": "table",
    "source": "documento.pdf",
    "page": 2,
    # ... altri metadati
  }
}
```

### 🖼️ Immagini

```python
# Ogni immagine ora include:
{
  "page_content": "Immagine: 1\nImage's Description: [descrizione AI]...",
  "metadata": {
    "image_number": 1,
    "type": "image", 
    "source": "documento.pdf",
    "page": 3,
    # ... altri metadati
  }
}
```

## 🔄 Funzionamento

### Contatori Globali
```python
# Inizializzazione nel parser
global_image_counter = 0
global_table_counter = 0

# Incremento per ogni elemento trovato
global_image_counter += 1  # Per ogni immagine accettata
global_table_counter += 1  # Per ogni tabella estratta
```

### Contenuto per Embedding
```python
# TABELLE: Include numerazione + contenuto + contesto
numbered_table_content = f"Tabella: {global_table_counter}\n{enhanced_table_content}"

# IMMAGINI: Include numerazione + descrizioni AI + contesto
caption_parts = [
    f"Immagine: {global_image_counter}",
    f"Image's Description: {ai_caption}",
    f"Image's Text: {ocr_text}",
    # ... altre informazioni
]
```

## 🎯 Vantaggi

### Query Semantiche Avanzate
Ora funzionano query come:
- "mostra la tabella 3"
- "contenuto della quinta tabella"
- "cosa contiene l'immagine numero 2"
- "seconda immagine del documento"

### Ricerca Ibrida
- **Semantica**: Via embedding del contenuto numerato
- **Diretta**: Via metadati `table_number` e `image_number`
- **Combinata**: Filtri sui metadati + ricerca semantica

### Tracciabilità
- Numerazione sequenziale garantisce ordine di apparizione
- Log dettagliati mostrano range di numerazione
- Metadati permettono mapping preciso documento → elementi

## 📊 Test e Validazione

### Script di Test
```bash
# Test numerazione globale completa
python scripts/test_global_numbering.py

# Test specifico per tabelle (quando disponibili)
python scripts/test_table_numbering.py
```

### Risultati Verificati
```
✅ Numerazione sequenziale: 1, 2, 3, ...
✅ Numero incluso nell'embedding: "Immagine: 1\n..."
✅ Numero nei metadati: {"image_number": 1}
✅ Log informativi: "numerazione 1-3"
```

## 💡 Utilizzo nell'Indexer

Il sistema di embedding utilizza automaticamente il contenuto numerato:

```python
# Tabelle
table_texts = [table.table_markdown for table in table_elements]
# Ora include: "Tabella: 1\n[contenuto]"

# Immagini  
image_texts = [image.page_content for image in image_elements]
# Ora include: "Immagine: 1\nImage's Description: ..."
```

## 🔍 Query Examples

### Retrieval per Numero Specifico
```python
# Via metadati
results = qdrant_client.search(
    collection_name="documents",
    query_vector=embedding,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="table_number",
                match=models.MatchValue(value=3)
            )
        ]
    )
)

# Via embedding semantico
results = qdrant_client.search(
    collection_name="documents", 
    query_vector=embed_query("tabella numero 3"),
    limit=5
)
```

### Query Naturali
- "Mostra la prima tabella" → Trova `table_number: 1`
- "Immagine due" → Trova `image_number: 2` 
- "Terza figura del documento" → Ricerca semantica + numero

## 📈 Impatto

### Prima
- Query: "tabella 3" → Risultati casuali
- Embedding: Solo contenuto senza contesto numerico
- Metadati: Nessuna numerazione

### Dopo  
- Query: "tabella 3" → Risultato preciso
- Embedding: "Tabella: 3\n[contenuto...]"
- Metadati: `{"table_number": 3}`
- Logging: "tabelle (numerazione 1-5)"

La numerazione globale trasforma il sistema da una ricerca puramente semantica a una ricerca **semantica + strutturale**, permettendo query precise e tracciabilità completa degli elementi multimodali nel documento.
