from pathlib import Path

RAW_DATA_PATH = Path('data/raw.csv')

def load_data(path=RAW_DATA_PATH):
    import pandas as pd
    return pd.read_csv(path)

if __name__ == '__main__':
    df = load_data()
    print(df.head())

