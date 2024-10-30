# 🧠 Sentiment Analysis on Customer Feedback

Questo è un progetto di **Analisi dei Sentimenti su Feedback dei Clienti**. Sfrutta modelli di machine learning avanzati per classificare automaticamente i feedback dei clienti in positivi, negativi o neutri. È pensato per aiutare le aziende a raccogliere informazioni dai commenti dei clienti, per migliorare i prodotti o i servizi offerti.

## 🔍 Descrizione del Progetto

Utilizzo un modello di deep learning basato su **DistilBERT** per l'elaborazione del linguaggio naturale (NLP), ottimizzato per rilevare il sentimento nei testi. Il modello è stato addestrato sul dataset **Sentiment140**, che comprende oltre un milione di tweet etichettati, rendendolo una base solida per l'analisi dei sentimenti. 

### ✨ Funzionalità Principali
- **Pipeline di Preprocessing**: Pulizia e preparazione dei dati per l'addestramento.
- **Modello di NLP Ottimizzato**: Utilizza un modello ridotto ma efficiente per gestire grandi volumi di dati.
- **Valutazione delle Performance**: Calcola metriche di accuratezza, precisione e F1-score per valutare il modello.
- **Analisi Interattiva dei Feedback**: Uno script permette di analizzare nuovi commenti, indicando rapidamente il sentimento associato.

## 📂 Struttura del Progetto

- **📁 data/**: Contiene i file CSV con i dati grezzi e processati.
- **🧠 models/**: Directory per salvare il modello addestrato.
- **📓 notebooks/**: Notebook Jupyter per l'esplorazione dei dati.
- **📜 src/**: Codice sorgente, suddiviso in:
  - `train_sentiment_model.py`: Script per l'addestramento del modello.
  - `analyze_feedback.py`: Script per l'analisi interattiva dei feedback.
  - `data_preprocessing.py`: Funzioni di pulizia e preprocessamento dei dati.
- **📊 results/**: Cartella per i risultati dell'addestramento e i log.

## 🛠️ Tecnologie Utilizzate

- **Python**
- **Transformers di Hugging Face**
- **Pandas** e **NumPy** per la manipolazione dei dati
- **scikit-learn** per le metriche di valutazione
- **PyTorch** per l'addestramento del modello

## 🚀 Come Iniziare

### Prerequisiti

- Python 3.8+
- Git LFS per gestire i file di grandi dimensioni

### Installazione

1. **Clona il repository**:
   ```bash
   git clone https://github.com/tuo_username/sentiment_analysis.git
   cd sentiment_analysis
   ```

2. **Installa le dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Scarica il dataset Sentiment140**:
   Segui le istruzioni per scaricare il dataset da [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) e salvalo nella cartella `data/`.

### ⚙️ Addestramento del Modello

Esegui il comando seguente per addestrare il modello:
```bash
python src/train_sentiment_model.py
```

### 💬 Analisi dei Feedback

Puoi analizzare singoli feedback in modo interattivo:
```bash
python src/analyze_feedback.py
```

## 📈 Risultati e Valutazione

Il modello raggiunge un'accuratezza di circa **85%** sul dataset di test di Sentiment140. Le metriche di valutazione (accuratezza, F1-score, precisione e richiamo) sono salvate nella cartella `results/` dopo l'addestramento.

## 🌟 Prospettive Future

- **Adattamento a Domini Specifici**: Personalizzare il modello per feedback su prodotti specifici, recensioni di app, ecc.
- **Miglioramento del Preprocessing**: Implementare tecniche avanzate come lemmatizzazione e rimozione di stopword.
- **Integrazione con una API REST**: Creare un’API per offrire analisi dei sentimenti in tempo reale.