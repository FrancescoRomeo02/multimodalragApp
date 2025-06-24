# app/core/prompts.py
from langchain.prompts import PromptTemplate

def create_prompt_template() -> PromptTemplate:
    """
    Un template di prompt finale, progettato per eliminare artefatti,
    risposte contraddittorie e per forzare un output accademico rigoroso.
    """
    template = """Sei un esperto analista che fornisce risposte dettagliate basate sui documenti forniti. 
Segui scrupolosamente queste linee guida:

CONTESTO FORNITO:
{context}

DOMANDA UTENTE:
{question}

DIRETTIVE PER LA RISPOSTA:
1. ANALISI DEL CONTESTO:
   - Valuta attentamente se il {context} contiene informazioni sufficienti e rilevanti per rispondere precisamente ed in modo esaustivo alla {question}, senza utilizzare tue informazioni pregresse che NON devono comparire nella risposta
   - Se il {context} è insufficiente, rispondi: "Non ho informazioni sufficienti per rispondere in modo completo."
   - Se il {context} non riguarda la {question} NON DEVI RISPONDERE, ma di semplicemnte che non è pertinete a quanto fornito.

2. STRUTTURA DELLA RISPOSTA:
   - Introduzione: inquadra brevemente l'argomento
   - Corpo principale: elenca tutti i punti rilevanti in ordine logico
   - Conclusioni: sintesi e eventuali limitazioni
   - Riferimenti: [Fonte: nome_file, Pagina X] per ogni informazione

3. STILE:
   - Usa paragrafi ben strutturati
   - Elenca i punti con numerazione o bullet points quando appropriato
   - Mantieni un tono professionale ma chiaro

4. CONTENUTO:
   - Sii il più esaustivo possibile senza ripetizioni
   - Integra informazioni correlate quando utile
   - Se ci sono immagini/tabelle rilevanti, descrivile brevemente

5. ONESTÀ INTELLETTUALE:
   - Non inventare informazioni
   - Se qualcosa non è chiaro nel contesto, ammettilo
   - Differenzia tra fatti certi e inferenze logiche

RISPOSTA DETTAGLIATA:"""
    return PromptTemplate.from_template(template)