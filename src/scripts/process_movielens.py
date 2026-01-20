import pandas as pd
import json
from pathlib import Path

# Define Paths
# Assuming the script is run from the project root or we can derive from __file__
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

raw_data_path = project_root / 'data' / 'raw' / 'movielens-1m' / 'movies.dat'
output_path = project_root / 'data' / 'processed' / 'movielens_metadata.jsonl'

def process_movielens_metadata():
    print(f"Reading from: {raw_data_path}")
    
    # MovieLens 1M movies.dat format: MovieID::Title::Genres
    # Encoding is usually ISO-8859-1 (latin-1)
    
    try:
        # Using python engine for '::' separator regex support
        df = pd.read_csv(
            raw_data_path, 
            sep='::', 
            engine='python', 
            header=None, 
            names=['MovieID', 'Title', 'Genres'],
            encoding='ISO-8859-1'
        )
        
        print(f"Loaded {len(df)} movies.")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                item = {
                    'id': str(row['MovieID']), # Convert to string to match RecSys ID types roughly
                    'title': row['Title'],
                    'genres': row['Genres'].split('|')
                }
                f.write(json.dumps(item) + '\n')
                
        print(f"Successfully wrote metadata to {output_path}")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    process_movielens_metadata()
