from src.sentiment_analysis import analyze_sentiment

def main():
    print("Analisi dei Sentimenti per Feedback dei Prodotti")
    while True:
        feedback = input("Inserisci il feedback ('exit' per uscire): ")
        if feedback.lower() == 'exit':
            break
        sentiment = analyze_sentiment(feedback)
        print(f"Sentimento: {sentiment}")

if __name__ == "__main__":
    main()
