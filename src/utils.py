import pandas as pd

def load_dataset(file_path):
    return pd.read_csv(file_path)

def save_results(results, output_file):
    results.to_csv(output_file, index=False)
    print(f"Risultati salvati in {output_file}")
