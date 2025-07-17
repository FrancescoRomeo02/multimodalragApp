# Upload Sequenziale - Guida Utente

## Nuova Funzionalità di Caricamento Multiplo

L'interfaccia è stata aggiornata per supportare il **caricamento sequenziale** di più file PDF con un ritardo configurabile di 30 secondi tra ogni caricamento.

## Come Utilizzare la Funzionalità

### 1. Selezione dei File
- Nel widget "Upload Documents" nella sidebar, ora puoi selezionare **più file PDF contemporaneamente**
- Clicca su "Choose PDF files" e seleziona tutti i file che vuoi caricare
- La lista dei file selezionati verrà mostrata con dimensioni e dettagli

### 2. Avvio del Caricamento Sequenziale
- Una volta selezionati i file, vedrai:
  - 📁 Numero di file selezionati
  - Elenco espandibile dei file con dimensioni
  - 🚀 Pulsante "Avvia Caricamento"
  - ⏱️ Indicazione del ritardo (30 secondi)
  - ⏰ Tempo stimato totale

### 3. Processo di Caricamento
Durante il caricamento sequenziale:

- **Progress Bar**: Mostra il progresso generale (es. "File 2/5 completati")
- **File Corrente**: Indica quale file è attualmente in elaborazione
- **Countdown**: Mostra il tempo rimanente prima del prossimo file
- **Risultati**: Espandibile per vedere i file già processati
- **⏹️ Pulsante Ferma**: Per interrompere il processo in qualsiasi momento

### 4. Monitoraggio in Tempo Reale
- L'interfaccia si aggiorna automaticamente ogni 2 secondi
- Visualizzazione dello stato per ogni file:
  - ✅ Successo
  - ❌ Errore (con dettagli)
- Indicatore del tempo rimanente per il prossimo file

### 5. Completamento
Al termine del caricamento:

- 🎉 Messaggio di completamento
- **Statistiche finali**:
  - 📁 File Totali
  - ✅ Successi  
  - ❌ Fallimenti
- **Dettagli risultati**: Espandibile per vedere tutti i risultati
- **🔄 Nuovo Caricamento**: Per resettare e iniziare un nuovo processo
- **🔄 Riprova Fallimenti**: Per riprovare solo i file falliti (se presenti)

## Vantaggi del Caricamento Sequenziale

### 📊 **Controllo del Carico**
- Evita di sovraccaricare il sistema con troppi processi simultanei
- Ritardo di 30 secondi per permettere al sistema di elaborare completamente ogni file

### 🔍 **Tracciabilità**
- Monitoraggio dettagliato di ogni file
- Log degli errori per troubleshooting
- Timestamp per ogni operazione

### 🎯 **User Experience**
- Feedback visivo costante
- Possibilità di interrompere il processo
- Riprova automatica dei fallimenti

### ⚡ **Gestione Errori**
- Continua con i file successivi anche se uno fallisce
- Dettagli specifici per ogni errore
- Opzione per riprovare solo i file falliti

## Configurazione Tecnica

### Parametri Modificabili
```python
# In streamlit_app/components/sequential_uploader.py
SequentialUploader(delay_seconds=30)  # Modifica il ritardo
```

### Limiti e Considerazioni
- **Ritardo fisso**: 30 secondi tra ogni file
- **Auto-refresh**: L'interfaccia si aggiorna automaticamente
- **Memoria della sessione**: Lo stato persiste durante la sessione
- **File supportati**: Solo PDF

## Risoluzione Problemi

### Errori Comuni
1. **File troppo grande**: Controlla i limiti di dimensione
2. **Formato non supportato**: Assicurati che sia un PDF valido  
3. **Errore di rete**: Verifica la connessione al database
4. **Timeout**: Per file molto grandi, potrebbe essere necessario più tempo

### Debug
- Controlla i log nella sezione "Risultati"
- Usa "Riprova Fallimenti" per i file con errori
- I dettagli degli errori sono visibili nell'interfaccia

## Utilizzo Avanzato

### Caricamento in Background
- Il processo continua anche se navighi in altre sezioni
- Lo stato viene mantenuto nella sessione corrente
- Puoi interagire con la chat mentre i file vengono caricati

### Gestione di Grandi Quantità
Per caricare molti file (>10):
1. Dividi in gruppi più piccoli
2. Usa il caricamento sequenziale per ogni gruppo
3. Monitora la memoria del sistema
4. Considera di aumentare il ritardo per file molto grandi

---

## Supporto Tecnico

Per problemi o domande:
- Controlla i log dell'applicazione
- Verifica la connessione al database Qdrant
- Assicurati che i servizi backend siano attivi
