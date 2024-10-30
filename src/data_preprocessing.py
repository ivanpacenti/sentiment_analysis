import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Rimuove punteggiatura
    text = re.sub(r'\s+', ' ', text)  # Rimuove spazi multipli
    return text.strip()

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)
    df['feedback_text'] = df['feedback_text'].apply(clean_text)
    df.to_csv(output_file, index=False)
    print("Dati preprocessati e salvati.")

# Esegui il preprocessing
preprocess_data('../data/raw_data.csv', '../data/processed_data.csv')
