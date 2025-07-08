# Data Directory

Questa directory contiene tutti i dati utilizzati dal progetto MultimodalRAG.

## Struttura

- **`raw/`**: PDF e documenti originali da processare
- **`processed/`**: Dati elaborati e embeddings cache
- **`models/`**: Modelli pre-addestrati (YOLO, embedding models)
- **`temp/`**: File temporanei di processing

## Note

- I file in `temp/` vengono automaticamente puliti
- I modelli in `models/` possono essere grandi (esclusi da git se necessario)
- `raw/` contiene i documenti di input dell'utente
