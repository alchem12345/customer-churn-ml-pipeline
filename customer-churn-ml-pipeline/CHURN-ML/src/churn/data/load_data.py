import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
    return df