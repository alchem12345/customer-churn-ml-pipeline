import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Total_charges'] = pd.to_numeric(df['Total_charges'],errors='coerce')
    return df