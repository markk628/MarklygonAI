import pandas as pd
from pathlib import Path

def convert_raw_json_to_pkl(file_name_without_extension: str):
    """
    Converts a raw JSON file for a given file name to PKL format.
    The JSON file is expected to be in DATA_DIR / 'raw' / '1_minute'.
    The PKL file will be saved in the same directory.

    Args:
        file_name_without_extension (str): The base name of the file (e.g., 'PLTR').
    """
    try:
        # Attempt to import DATA_DIR from src.config.config
        # This assumes that your project structure and PYTHONPATH are set up correctly
        # for this script to find src.config.config
        from src.config.config import DATA_DIR
        print(f"Successfully imported DATA_DIR: {DATA_DIR}")
    except ImportError:
        # Fallback DATA_DIR if import fails
        # Adjust the path as needed if running this script standalone or if src.config.config is not accessible
        print("Failed to import DATA_DIR from src.config.config. Using a fallback relative path './data'.")
        print("Ensure this script is run from the project root or adjust the DATA_DIR path accordingly.")
        # Assuming the script is run from the root of MarklygonAI project
        DATA_DIR = Path('.') / 'data'
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created fallback DATA_DIR at {DATA_DIR.resolve()}")

    raw_data_sub_dir = Path('raw') / '1_minute'
    json_file_path = DATA_DIR / raw_data_sub_dir / f'{file_name_without_extension.upper()}.json'
    pkl_file_path = DATA_DIR / raw_data_sub_dir/'pkl' / f'{file_name_without_extension.upper()}.pkl'

    print(f"Processing file: {file_name_without_extension}")
    print(f"Expected JSON file path: {json_file_path.resolve()}")
    print(f"Output PKL file path: {pkl_file_path.resolve()}")

    # Create the target directory if it doesn't exist
    output_dir = DATA_DIR / raw_data_sub_dir
    if not output_dir.exists():
        print(f"Output directory {output_dir.resolve()} does not exist. Creating it...")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {output_dir.resolve()}")

    # Check if the JSON file exists
    if not json_file_path.exists():
        print(f"Error: JSON file not found at {json_file_path.resolve()}. Please ensure the file exists.")
        return

    try:
        print(f"Reading JSON file: {json_file_path.resolve()}...")
        # JSON file is a single array of records, so lines=True should be removed.
        df = pd.read_json(json_file_path, orient='records')
        
        if df.empty:
            print(f"Warning: The JSON file {json_file_path.resolve()} is empty or could not be parsed into a non-empty DataFrame. A PKL file will be created but may also be empty.")

        print(f"Saving DataFrame to PKL file: {pkl_file_path.resolve()}...")
        df.to_pickle(pkl_file_path)
        print(f"Successfully converted '{json_file_path.name}' to '{pkl_file_path.name}' at {pkl_file_path.resolve()}")

    except ValueError as ve:
        print(f"Error: Could not parse JSON file {json_file_path.resolve()}. It might not be in a valid format for pd.read_json or might be empty/corrupted. Details: {ve}")
        print("Ensure the JSON file is a valid array of records.")
    except pd.errors.EmptyDataError: # Though read_json might raise ValueError for empty/invalid JSON
        print(f"Error: The JSON file {json_file_path.resolve()} is effectively empty or unparsable by pandas.")
    except Exception as e:
        print(f"An unexpected error occurred during conversion for {file_name_without_extension}: {e}")

if __name__ == '__main__':
    # Example usage:
    # Change 'PLTR' to the base name of your JSON file (without .json extension)
    files_to_process = ['NVDA']  # e.g., if you have PLTR.json, SOFI.json, put ['PLTR', 'SOFI']
    
    print("Starting JSON to PKL conversion process...")
    for file_base_name in files_to_process:
        convert_raw_json_to_pkl(file_base_name)
        print("-" * 30) 
    print("JSON to PKL conversion process finished.") 