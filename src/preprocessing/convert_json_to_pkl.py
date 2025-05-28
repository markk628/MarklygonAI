import pandas as pd
from pathlib import Path
from src.config.config import DATA_DIR
from src.utils.utils import create_directory

def convert_json_to_pkl(ticker: str, output_dir: str):
    create_directory(output_dir)
    ticker_file_path = DATA_DIR / 'raw' / '1_minute' / f'{ticker.upper()}.json'
    pd.read_json(ticker_file_path).to_pickle(output_dir / f'{ticker.upper()}.pkl')

if __name__ == '__main__':
    # Example usage:
    # Change 'PLTR' to the base name of your JSON file (without .json extension)
    files_to_process = ['NVDA']  # e.g., if you have PLTR.json, SOFI.json, put ['PLTR', 'SOFI']
    
    print("Starting JSON to PKL conversion process...")
    for file_base_name in files_to_process:
        convert_raw_json_to_pkl(file_base_name)
        print("-" * 30) 
    print("JSON to PKL conversion process finished.")