from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model():
    # Carica il tokenizer e il modello addestrato dalla directory
    tokenizer = AutoTokenizer.from_pretrained("models/sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained("models/sentiment_model")
    return tokenizer, model

def analyze_feedback(feedback_text):
    # Carica il modello e il tokenizer
    tokenizer, model = load_model()

    # Preprocessa il testo di input
    inputs = tokenizer(feedback_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    # Ottieni le previsioni
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Mappa delle etichette di sentimento
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

    # Ritorna il sentimento predetto
    return sentiment_map[predicted_class]

if __name__ == "__main__":
    # Richiede l'input all'utente
    feedback = input("Inserisci il feedback da analizzare: ")
    sentiment = analyze_feedback(feedback)
    print(f"Sentimento previsto: {sentiment}")
