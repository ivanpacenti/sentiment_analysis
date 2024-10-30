from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("../models/sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained("../models/sentiment_model")
    return tokenizer, model

def analyze_sentiment(text):
    tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=1).item()
    return ["negative", "neutral", "positive"][sentiment]

# Test rapido
if __name__ == "__main__":
    feedback = "Amo questo prodotto!"
    print(f"Sentimento: {analyze_sentiment(feedback)}")
