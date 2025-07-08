# Guida Rapida - MultimodalRAG


## Setup

### Prerequisiti Minimi
- **Docker** installato ([Download Docker](https://docs.docker.com/get-docker/))
- **Chiave API Groq** gratuita ([Registrati qui](https://console.groq.com/keys))

### Passi per l'Avvio

1. **Clona il progetto**
   ```bash
   git clone <repository-url>
   cd multimodalrag
   ```

2. **Configura API Key**
   ```bash
   cp .env.example .env
   # Apri il file .env e sostituisci "your_groq_api_key_here" con la tua chiave
   ```

3. **Avvia tutto**
   ```bash
   docker-compose up -d
   ```

4. **Accedi all'applicazione**
   - Apri browser su: http://localhost:8501
   - L'app si avvierà automaticamente

## Test delle Funzionalità

### 1. Upload di un PDF
- Usa la sidebar sinistra per caricare un PDF
- Il sistema processerà automaticamente testo, immagini e tabelle

### 2. Query di Test
Prova queste query per testare le funzionalità:

**Query Generali:**
- "Di cosa parla questo documento?"
- "Riassumi i punti principali"

**Query su Immagini:**
- "Mostrami le immagini presenti"
- "Che cosa mostrano le figure?"

**Query su Tabelle:**
- "Quali dati ci sono nelle tabelle?"
- "Mostrami i risultati numerici"

**Query Multimodali:**
- "Combina informazioni da testo e immagini"

## Cosa Osservare

### Funzionalità Chiave
- ✅ **Upload PDF** - Processing automatico
- ✅ **Estrazione multimodale** - Testo, immagini, tabelle
- ✅ **Ricerca semantica** - Risposte contestuali
- ✅ **Riferimenti alle fonti** - Tracciabilità
- ✅ **Interface moderna** - UX pulita e intuitiva

### Aspetti Tecnici Avanzati
- **AI Vision** - Caption automatiche con BLIP
- **OCR** - Estrazione testo da immagini
- **Object Detection** - Riconoscimento oggetti con YOLO
- **Database vettoriale** - Qdrant per similarità semantica
- **LLM Integration** - Groq API per generazione risposte

## Risoluzione Problemi

### Errore "Qdrant not reachable"
```bash
# Verifica che Qdrant sia attivo
docker ps | grep qdrant

# Se non è attivo, riavvia tutto
docker-compose down
docker-compose up -d
```

### Errore "Invalid API Key"
```bash
# Verifica che la chiave sia corretta nel file .env
cat .env | grep GROQ_API_KEY

# La chiave deve iniziare con "gsk_"
```

### Porta già in uso
```bash
# Cambia porta nel docker-compose.yml
# Cerca la riga "8501:8501" e cambiala in "8502:8501"
# Poi accedi su http://localhost:8502
```

## Metriche e Monitoring

Per vedere le performance in tempo reale:

1. **Abilita monitoring** nel file `.env`:
   ```bash
   ENABLE_PERFORMANCE_MONITORING=true
   ```

2. **Avvia con Grafana**:
   ```bash
   docker-compose --profile monitoring up -d
   ```

3. **Accedi ai dashboard**:
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090


## Supporto

Se hai problemi durante l'esecuzione:

1. **Controlla i log**:
   ```bash
   docker-compose logs -f multimodal-rag
   ```

2. **Reset completo**:
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

3. **Setup locale alternativo**:
   ```bash
   pip install -r requirements.txt
   docker run -d -p 6333:6333 qdrant/qdrant
   streamlit run streamlit_app/Home.py
   ```
---
