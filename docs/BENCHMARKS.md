# RAG Benchmark Analysis - Sistema di Analisi Correttezza

Questo progetto implementa un sistema completo per l'analisi dei risultati di benchmark RAG, incluso il calcolo del tasso di correttezza del modello locale rispetto a un modello di riferimento (Morphik), utilizzando SBERT per confronti semantici e identificazione di aree di fallimento e successo.

## 📁 Struttura del Progetto

```
bench_script/
├── bench_res/                         # Directory con i file JSON di benchmark
│   ├── rag_bench_pd1.json            # Risultati benchmark documento 1
│   ├── rag_bench_pd2.json            # Risultati benchmark documento 2
│   ├── rag_bench_pd3.json            # Risultati benchmark documento 3
│   └── ... (altri file benchmark)
├── bench_file/                        # Directory con i PDF originali
│   ├── pdf_01.pdf
│   ├── pdf_02.pdf
│   └── ... (documenti fonte)
├── plots/                             # Grafici di analisi generati
│   ├── 01_performance_per_difficolta.png      # Performance vs difficoltà
│   ├── 02_performance_per_argomento.png       # Performance per argomento
│   ├── 03_heatmap_argomento_difficolta.png    # Interazione difficoltà-argomento
│   ├── 04_distribuzione_performance.png       # Distribuzione generale
│   ├── 05_correlazione_chunk_performance.png  # Correlazione retrieval-risposta
│   ├── 06_lunghezza_per_qualita.png          # Analisi lunghezza risposte
│   ├── 07_performance_per_tipo_domanda.png    # Performance per tipo domanda
│   ├── 08_coverage_termini_tecnici.png       # Coverage terminologia
│   ├── 09_casi_non_risposta.png              # Fallimenti totali
│   ├── 10_peggiori_performance.png           # Peggiori performance effettive
│   ├── 11_distribuzione_per_difficolta.png   # Boxplot per difficoltà
│   ├── 12_performance_per_paper.png          # Performance per documento*
│   └── 13_variabilita_per_paper.png          # Variabilità per documento*
├── correctness_analysis.py            # Script principale di analisi
├── correctness_analysis_detailed.csv  # Risultati dettagliati esportati
├── CORRECTNESS_REPORT.md             # Report markdown generato
├── GUIDA_GRAFICI_COMPLETA.md         # Guida completa ai grafici
└── README.md                         # Questo file

* = Generati solo con più documenti
```

## 🚀 Installazione e Setup

1. **Preparazione ambiente:**
   ```bash
   cd bench_script
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # oppure su Windows: .venv\Scripts\activate
   ```

2. **Installazione dipendenze:**
   ```bash
   pip install sentence-transformers numpy pandas matplotlib seaborn scikit-learn
   ```

## 📊 Utilizzo

### Analisi Completa (Raccomandato)
```bash
python correctness_analysis.py
```

**Il script esegue automaticamente:**
- ✅ Analisi della qualità delle risposte tramite SBERT
- ✅ Classificazione in livelli di correttezza
- ✅ Identificazione di pattern di fallimento e successo  
- ✅ Generazione di 13 grafici individuali
- ✅ Report dettagliato in markdown

**Output generati:**
- `correctness_analysis_detailed.csv` - Dati dettagliati per ogni domanda
- `CORRECTNESS_REPORT.md` - Report completo con metriche e raccomandazioni
- `plots/01_*.png` fino a `plots/13_*.png` - Grafici di analisi individuali

### Interpretazione dei Risultati

**Consultare i grafici in questo ordine:**
1. `04_distribuzione_performance.png` - Overview generale del sistema
2. `01_performance_per_difficolta.png` - Gestione della complessità
3. `02_performance_per_argomento.png` - Domini problematici
4. `05_correlazione_chunk_performance.png` - Diagnosi retrieval vs generazione
5. `09_casi_non_risposta.png` - Fallimenti critici

**Per guida dettagliata:** Consultare `GUIDA_GRAFICI_COMPLETA.md`

## 📊 Formato Dati di Input

### Struttura JSON Richiesta:
```json
{
  "file": "nome_paper.pdf",
  "questions": [
    {
      "question_id": "q1",
      "question": "Testo della domanda",
      "Morphik": {
        "response": "Risposta modello benchmark",
        "chunks": [0, 3, 1, 16, 5]  // Indicizzazione da 0
      },
      "local": {
        "response": "Risposta modello locale",
        "chunks": [1, 3, 5, 6, 11]  // Indicizzazione da 1 (auto-convertita)
      }
    }
  ]
}
```

## 📊 Metriche e Soglie

### Classificazione della Correttezza:
- **🏆 Eccellente**: ≥ 0.85 (Qualità molto alta)
- **✅ Buono**: ≥ 0.70 (Qualità accettabile) 
- **⚠️ Accettabile**: ≥ 0.50 (Necessita miglioramenti)
- **❌ Scarso**: ≥ 0.30 (Problematico)
- **🚫 Molto Scarso**: < 0.30 (Critico)

### KPI di Sistema:
- **Tasso Buono+**: Target ≥ 50%
- **Similarità Media**: Target ≥ 0.60
- **Casi Critici**: Target ≤ 10%
- **Correlazione Retrieval**: Target ≥ 0.40

### Metriche Calcolate:
- **Similarità Semantica**: Confronto SBERT tra risposte
- **Qualità Retrieval**: F1, Precision, Recall sui chunk
- **Analisi Linguistica**: Lunghezza, terminologia tecnica
- **Pattern Analysis**: Tipo domanda, difficoltà, argomento

## 🎯 Checklist di Controllo Qualità

**✅ Sistema Sano:**
- [ ] Similarità media > 0.60
- [ ] Tasso Buono+ > 50%
- [ ] Correlazione retrieval-performance > 0.4
- [ ] Pochi casi di non-risposta (< 5%)

**⚠️ Richiede Attenzione:**
- [ ] Similarità media 0.40-0.60
- [ ] Alcuni argomenti problematici
- [ ] Alta variabilità tra documenti

**🚨 Critico:**
- [ ] Similarità media < 0.40
- [ ] Molti casi di non-risposta
- [ ] Nessuna correlazione retrieval-performance

## 🔧 Troubleshooting

### Problemi Comuni:
1. **Dipendenze mancanti**: Installare tutte le librerie richieste
2. **File benchmark non trovati**: Verificare la struttura della cartella `bench_res/`
3. **Grafici non generati**: Controllare i permessi della cartella `plots/`

### Debug:
```python
# Per debug dettagliato, modificare nel codice:
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📚 Documentazione Aggiuntiva

- **`GUIDA_GRAFICI_COMPLETA.md`** - Guida dettagliata per interpretare tutti i grafici
- **`CORRECTNESS_REPORT.md`** - Report automaticamente generato con risultati e raccomandazioni
- **`correctness_analysis_detailed.csv`** - Dati grezzi per analisi personalizzate

---

*Sistema di analisi RAG sviluppato per valutazione comparativa delle performance di modelli locali vs benchmark*
