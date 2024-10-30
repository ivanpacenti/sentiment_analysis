from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def load_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None,
                     names=["polarity", "id", "date", "query", "user", "text"])

    # Mappo la colonna 'polarity' su sentimenti numerici (0 = negativo, 1 = neutro, 2 = positivo)
    label_map = {0: 0, 2: 1, 4: 2}
    df['labels'] = df['polarity'].map(label_map)

    df = df[['text', 'labels']]
    df.rename(columns={'text': 'feedback_text'}, inplace=True)

    # Converte il DataFrame in un Dataset di Hugging Face
    dataset = Dataset.from_pandas(df)
    return dataset


def preprocess_function(examples):
    return tokenizer(examples["feedback_text"], truncation=True, padding="max_length", max_length=128)


def train_model():

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)


    dataset = load_data('../data/training.1600000.processed.noemoticon.csv')
    dataset = dataset.train_test_split(test_size=0.01)


    tokenized_train = dataset["train"].map(preprocess_function, batched=True)
    tokenized_test = dataset["test"].map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test
    )
    trainer.train()

    model.save_pretrained("../models/sentiment_model")
    tokenizer.save_pretrained("../models/sentiment_model")


if __name__ == "__main__":
    train_model()
