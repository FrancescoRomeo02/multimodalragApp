# 📊 GUIDA COMPLETA AI GRAFICI DI ANALISI RAG

Questa guida spiega tutti i grafici generati dal sistema di analisi delle performance del modello RAG locale rispetto a Morphik. Ogni grafico è ora salvato in un file separato per una migliore organizzazione e consultazione.

---

## 📈 GRAFICI DI PERFORMANCE GENERALE

### 📊 `01_performance_per_difficolta.png` - Performance per Livello di Difficoltà
- **Scopo**: Mostra come la performance del modello varia con la difficoltà delle domande (1=facile → 10=difficile)
- **Cosa osservare**: 
  - Trend generale: il modello dovrebbe performance meglio su domande facili
  - Valori anomali: difficoltà medie che performano meglio di quelle facili
  - Deviazione standard: indica la consistenza del modello per ogni livello
- **Interpretazione**: Barre più alte = migliore performance. Errori più piccoli = maggiore consistenza.

### 📊 `02_performance_per_argomento.png` - Performance per Macro-Argomento  
- **Scopo**: Identifica gli argomenti su cui il modello eccelle o fallisce
- **Categorie**: Energia, Tecnologia, Geografia, Matematica, Ambiente, Ingegneria, Economia, Fisica, Generale
- **Cosa osservare**:
  - Argomenti più problematici (barre corte)
  - Argomenti di forza del modello (barre lunghe)
  - Numero di domande per argomento (n=X)
- **Interpretazione**: Aiuta a identificare domini di conoscenza da potenziare.

### 📊 `03_heatmap_argomento_difficolta.png` - Heatmap Difficoltà vs Macro-Argomento
- **Scopo**: Visualizza l'interazione tra difficoltà e argomento
- **Colori**: Verde = buona performance, Rosso = performance scarsa
- **Cosa osservare**:
  - Celle rosse: combinazioni problematiche difficoltà-argomento
  - Celle verdi: combinazioni di successo
  - Pattern diagonali o specifici
- **Interpretazione**: Identifica se certi argomenti sono difficili solo ad alti livelli di difficoltà.

### 📊 `04_distribuzione_performance.png` - Distribuzione Performance Globale
- **Scopo**: Mostra la distribuzione complessiva delle performance con soglie di qualità
- **Linee di riferimento**:
  - Verde: Soglia eccellente (≥0.85)
  - Arancione: Soglia buona (≥0.70)  
  - Rossa: Soglia critica (≥0.30)
  - Blu: Media del sistema
- **Interpretazione**: Forma della distribuzione indica se il modello è consistente o ha performance bimodali.

---

## 🔬 GRAFICI DI ANALISI DETTAGLIATA

### 📊 `05_correlazione_chunk_performance.png` - Correlazione Qualità Retrieval vs Performance
- **Scopo**: Valuta se un migliore retrieval dei chunk porta a migliori risposte
- **Assi**: X = F1 Score del retrieval, Y = Similarità semantica della risposta
- **Linea rossa**: Tendenza di correlazione
- **Cosa osservare**:
  - Correlazione positiva forte: buon retrieval → buone risposte
  - Punti lontani dalla linea: casi anomali da investigare
  - Valore di correlazione mostrato
- **Interpretazione**: Alta correlazione indica che il problema è nel retrieval, bassa correlazione indica problemi nella generazione.

### 📊 `06_lunghezza_per_qualita.png` - Rapporto Lunghezza per Qualità Risposta
- **Scopo**: Analizza se risposte più lunghe/corte sono associate a migliore/peggiore qualità
- **Categorie**: Molto Scarso, Scarso, Accettabile, Buono, Eccellente
- **Boxplot**: Mostra distribuzione dei rapporti di lunghezza per ogni categoria
- **Cosa osservare**:
  - Mediana intorno a 1.0 = lunghezza simile a Morphik
  - Valori > 1.5 = risposte troppo prolisse
  - Valori < 0.5 = risposte troppo brevi
- **Interpretazione**: Aiuta a calibrare la lunghezza ideale delle risposte.

### 📊 `07_performance_per_tipo_domanda.png` - Performance per Tipo di Domanda
- **Scopo**: Identifica tipologie di domande problematiche
- **Tipi**: Definitoria, Comparativa, Procedurale, Analitica, Enumerativa, Altra
- **Cosa osservare**:
  - Tipi con performance basse
  - Numero di domande per tipo
  - Variabilità all'interno del tipo
- **Interpretazione**: Suggerisce strategie di prompting specifiche per tipo.

### 📊 `08_coverage_termini_tecnici.png` - Coverage Termini Tecnici per Argomento
### 📊 `08_coverage_termini_tecnici.png` - Coverage Termini Tecnici per Argomento
- **Scopo**: Valuta se il modello usa la terminologia tecnica appropriata
- **Metrica**: Rapporto tra termini tecnici nella risposta locale vs Morphik
- **Cosa osservare**:
  - Argomenti con bassa coverage tecnica
  - Variabilità (errori grandi = inconsistenza)
  - Valori > 1.0 = uso eccessivo di termini tecnici
- **Interpretazione**: Indica se il modello manca di precisione terminologica.

---

## 🚨 GRAFICI DI ANALISI CASI CRITICI

### 📊 `09_casi_non_risposta.png` - Casi di Non-Risposta  
- **Scopo**: Identifica quando il modello non riesce a fornire alcuna risposta (similarità = 0)
- **Visualizzazione**: Conteggio per macro-argomento dei casi di fallimento totale
- **Cosa osservare**:
  - Argomenti dove il modello fallisce completamente
  - Pattern di non-risposta per identificare problemi strutturali
  - Volume totale di fallimenti
- **Interpretazione**: Indica problemi critici nel retrieval o nella generazione che impediscono qualsiasi output.

### 📊 `10_peggiori_performance.png` - Peggiori Performance Effettive
- **Scopo**: Identifica i casi con le performance più basse tra quelli che hanno effettivamente risposto
- **Esclusioni**: Non include i casi di non-risposta (similarità = 0)
- **Cosa osservare**:
  - Performance molto basse ma non nulle
  - Pattern comuni nelle risposte scadenti
  - Confronto con soglie di accettabilità
- **Interpretazione**: Mostra dove il modello risponde ma con qualità molto scarsa, suggerendo problemi di generazione piuttosto che retrieval.

### 📊 `11_distribuzione_per_difficolta.png` - Distribuzione per Difficoltà
- **Scopo**: Boxplot che mostra la variabilità per categoria di difficoltà
- **Categorie**: Facile (1-3), Media (4-7), Difficile (8-10)
- **Cosa osservare**:
  - Mediane decrescenti con difficoltà
  - Outliers (punti esterni ai box)
  - Sovrapposizione tra categorie
- **Interpretazione**: Valuta se la classificazione di difficoltà è significativa e se il modello gestisce adeguatamente la complessità.

---

## 📄 GRAFICI DI ANALISI PER DOCUMENTO

*Generati solo se ci sono più documenti*

### 📊 `12_performance_per_paper.png` - Performance per Paper/Documento
- **Scopo**: Confronta la performance del modello su diversi documenti
- **Cosa osservare**:
  - Documenti particolarmente problematici
  - Variabilità tra documenti
  - Numero di domande per documento
- **Interpretazione**: Alcuni documenti potrebbero essere più difficili da processare.

### 📊 `13_variabilita_per_paper.png` - Variabilità Performance per Paper
- **Scopo**: Boxplot della consistenza all'interno di ogni documento
- **Cosa osservare**:
  - Box larghi = alta variabilità interna
  - Box stretti = performance consistente
  - Outliers all'interno dei documenti
- **Interpretazione**: Indica se alcuni documenti hanno sezioni più difficili.

---

## 🎯 COME UTILIZZARE I GRAFICI

### 1. **Workflow di Analisi Consigliato**:
   1. **Inizia con i grafici di performance generale** (01-04) per avere un quadro d'insieme
   2. **Approfondisci con l'analisi dettagliata** (05-08) per identificare le cause specifiche
   3. **Esamina i casi critici** (09-11) per comprendere i fallimenti
   4. **Se hai multipli documenti**, consulta i grafici per documento (12-13)

### 2. **Ordine di Consultazione Raccomandato**:
   - `04_distribuzione_performance.png` → Overview generale del sistema
   - `01_performance_per_difficolta.png` → Verifica gestione della complessità
   - `02_performance_per_argomento.png` → Identifica domini problematici
   - `05_correlazione_chunk_performance.png` → Diagnosi retrieval vs generazione
   - `09_casi_non_risposta.png` → Identifica fallimenti critici
   - `07_performance_per_tipo_domanda.png` → Strategie di miglioramento

### 3. **Indicatori di Alert**:
   - **🔴 Alert Rosso**: Similarità media < 0.30 in qualsiasi categoria
   - **🟡 Alert Giallo**: Similarità media < 0.50 in categorie principali  
   - **✅ Target**: Similarità media > 0.70 in tutte le categorie
   - **🏆 Eccellenza**: Similarità media > 0.85

---

## 🚨 DASHBOARD 3: Analisi Casi Critici (`critical_cases_analysis.png`)

### Grafico 1: Casi di Non-Risposta  
- **Scopo**: Identifica quando il modello non riesce a fornire alcuna risposta (similarità = 0)
- **Visualizzazione**: Conteggio per macro-argomento dei casi di fallimento totale
- **Cosa osservare**:
  - Argomenti dove il modello fallisce completamente
  - Pattern di non-risposta per identificare problemi strutturali
  - Volume totale di fallimenti
- **Interpretazione**: Indica problemi critici nel retrieval o nella generazione che impediscono qualsiasi output.

### Grafico 2: Peggiori Performance Effettive
- **Scopo**: Identifica i casi con le performance più basse tra quelli che hanno effettivamente risposto
- **Esclusioni**: Non include i casi di non-risposta (similarità = 0)
- **Cosa osservare**:
  - Performance molto basse ma non nulle
  - Pattern comuni nelle risposte scadenti
  - Confronto con soglie di accettabilità
- **Interpretazione**: Mostra dove il modello risponde ma con qualità molto scarsa, suggerendo problemi di generazione piuttosto che retrieval.

### Grafico 3: Distribuzione per Difficoltà
- **Scopo**: Boxplot che mostra la variabilità per categoria di difficoltà
- **Categorie**: Facile (1-3), Media (4-7), Difficile (8-10)
- **Cosa osservare**:
  - Mediane decrescenti con difficoltà
  - Outliers (punti esterni ai box)
  - Sovrapposizione tra categorie
- **Interpretazione**: Valuta se la classificazione di difficoltà è significativa e se il modello gestisce adeguatamente la complessità.

---

## 📄 DASHBOARD 4: Analisi per Documento (`paper_analysis.png`)

*Generata solo se ci sono più documenti*

### Grafico 1: Performance per Paper/Documento
- **Scopo**: Confronta la performance del modello su diversi documenti
- **Cosa osservare**:
  - Documenti particolarmente problematici
  - Variabilità tra documenti
  - Numero di domande per documento
- **Interpretazione**: Alcuni documenti potrebbero essere più difficili da processare.

### Grafico 2: Variabilità Performance per Paper
- **Scopo**: Boxplot della consistenza all'interno di ogni documento
- **Cosa osservare**:
  - Box larghi = alta variabilità interna
  - Box stretti = performance consistente
  - Outliers all'interno dei documenti
- **Interpretazione**: Indica se alcuni documenti hanno sezioni più difficili.

---

## 🎯 COME UTILIZZARE I GRAFICI

### 1. **Workflow di Analisi Consigliato**:
   1. Inizia con Dashboard 1 per overview generale
   2. Usa Dashboard 2 per analisi dettagliata delle cause
   3. Esamina Dashboard 3 per casi specifici
   4. Se multipli documenti, consulta Dashboard 4

### 2. **Indicatori di Alert**:
   - **🔴 Alert Rosso**: Similarità media < 0.30 in qualsiasi categoria
   - **🟡 Alert Giallo**: Similarità media < 0.50 in categorie principali  
   - **✅ Target**: Similarità media > 0.70 in tutte le categorie
   - **🏆 Eccellenza**: Similarità media > 0.85

### 4. **Pattern da Investigare**:
   - **Correlazione retrieval-performance bassa**: Problema nella generazione
   - **Performance peggiore su difficoltà alte**: Normale, ma verifica il gap
   - **Argomenti specifici problematici**: Mancanza di knowledge domain
   - **Alta variabilità**: Modello inconsistente, serve regularization

### 5. **Azioni Raccomandate per Categoria**:
   - **Retrieval**: Migliorare chunking, embedding, ranking
   - **Generazione**: Ottimizzare prompt, temperatura, max_tokens
   - **Knowledge**: Aggiungere documenti specifici per argomenti deboli
   - **Consistency**: Implementare ensemble methods o post-processing

---

## 📋 CHECKLIST DI CONTROLLO QUALITÀ

**✅ Sistema Sano**:
- [ ] Similarità media > 0.60 (`04_distribuzione_performance.png`)
- [ ] Tasso Buono+ > 50% (verificare nel report)
- [ ] Correlazione retrieval-performance > 0.4 (`05_correlazione_chunk_performance.png`)
- [ ] Nessun argomento con similarità < 0.30 (`02_performance_per_argomento.png`)
- [ ] Pochi o zero casi di non-risposta (`09_casi_non_risposta.png`)

**⚠️ Sistema che Necessita Attenzione**:
- [ ] Similarità media 0.40-0.60
- [ ] Tasso Buono+ 30-50% 
- [ ] Alcuni argomenti problematici
- [ ] Alta variabilità tra documenti (`13_variabilita_per_paper.png`)

**🚨 Sistema Critico**:
- [ ] Similarità media < 0.40
- [ ] Tasso Buono+ < 30%
- [ ] Molti casi con similarità = 0 (`09_casi_non_risposta.png`)
- [ ] Nessuna correlazione retrieval-performance (`05_correlazione_chunk_performance.png`)

---

## 📁 STRUTTURA FILE GENERATI

```
plots/
├── 01_performance_per_difficolta.png      # Performance vs difficoltà
├── 02_performance_per_argomento.png       # Performance per argomento
├── 03_heatmap_argomento_difficolta.png    # Interazione difficoltà-argomento
├── 04_distribuzione_performance.png       # Distribuzione generale
├── 05_correlazione_chunk_performance.png  # Correlazione retrieval-risposta
├── 06_lunghezza_per_qualita.png          # Analisi lunghezza risposte
├── 07_performance_per_tipo_domanda.png    # Performance per tipo domanda
├── 08_coverage_termini_tecnici.png       # Coverage terminologia
├── 09_casi_non_risposta.png              # Fallimenti totali
├── 10_peggiori_performance.png           # Peggiori performance effettive
├── 11_distribuzione_per_difficolta.png   # Boxplot per difficoltà
├── 12_performance_per_paper.png          # Performance per documento*
└── 13_variabilita_per_paper.png          # Variabilità per documento*

* = Generati solo con più documenti
```

---

*Documentazione aggiornata per il sistema di analisi RAG con grafici separati*
