import pandas as pd
import json
import os
from tqdm import tqdm
from src import config
from src.config import RETRIEVED_DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH, PROMPTING_DATA_PATH

# --- CONFIGURATION ---
# Adjust these paths to match your folder structure
CANDIDATES_PATH = RETRIEVED_DATA_PATH / 'sports_outdoors_5core/reduced_candidates_non_zero_recall.csv'
TRAIN_PATH = PROCESSED_DATA_PATH / 'sports_and_outdoors_global_temporal_split/train.csv'
METADATA_PATH = RAW_DATA_PATH / 'meta_sports_and_outdoors.jsonl'
OUTPUT_PATH = PROMPTING_DATA_PATH / 'sports_outdoors_prompting-data.jsonl'

# Constraint: How many past interactions to use as context?
# Too many will confuse the LLM or exceed token limits.
HISTORY_LIMIT = 100


def load_metadata_lookup(filepath):
    """
    Reads the metadata JSONL and creates a dictionary mapping item_id -> {title, desc}.
    Optimized to only keep necessary fields in memory.
    """
    print(f"Loading metadata from {filepath}...")
    lookup = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing Metadata"):
                try:
                    data = json.loads(line)
                    # The user specified 'parent_asin' is the item_id matching candidates
                    item_id = data.get('parent_asin')

                    if not item_id:
                        continue

                    title = data.get('title', 'Unknown Title')

                    # Description is a list in the source; join it into a single string
                    desc_raw = data.get('description', [])
                    description = " ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw)

                    # Store only what we need
                    lookup[item_id] = {
                        'title': title,
                        'description': description
                    }
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {filepath}")
        return {}

    print(f"Loaded metadata for {len(lookup)} items.")
    return lookup


def get_user_histories(filepath):
    """
    Reads the training data to get user interaction history.
    Assumes columns: user_id, item_id, rating, timestamp
    """
    print(f"Loading interaction history from {filepath}...")

    # Read CSV without header
    try:
        df = pd.read_csv(filepath, names=['user_id', 'item_id', 'rating', 'timestamp'])
    except FileNotFoundError:
        print(f"Error: Train file not found at {filepath}")
        return {}

    # Sort by timestamp descending so we get the most recent interactions
    # (Assuming we want the LLM to know what the user bought *recently*)
    df = df.sort_values('timestamp', ascending=False)

    # Group by user and take the most recent 'HISTORY_LIMIT' items
    # Returns a dictionary: {user_id: [item_id1, item_id2, ...]}
    history_map = df.groupby('user_id')['item_id'].apply(lambda x: x.head(HISTORY_LIMIT).tolist()).to_dict()

    print(f"Loaded history for {len(history_map)} users.")
    return history_map


def main():
    # 1. Load the helper data
    meta_lookup = load_metadata_lookup(METADATA_PATH)
    history_map = get_user_histories(TRAIN_PATH)

    # 2. Load the candidates
    print(f"Loading candidates from {CANDIDATES_PATH}...")
    try:
        df_candidates = pd.read_csv(CANDIDATES_PATH)
    except FileNotFoundError:
        print(f"Error: Candidates file not found at {CANDIDATES_PATH}")
        return

    # Ensure candidates are sorted by rank (1, 2, 3...)
    df_candidates = df_candidates.sort_values(['user_id', 'rank'])

    # 3. Process each user and write to output
    print(f"Composing {OUTPUT_PATH}...")

    # Open output file
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:

        # Group dataframe by user to process one user at a time
        grouped = df_candidates.groupby('user_id')

        for user_id, group in tqdm(grouped, desc="Generating Prompts"):

            # --- A. Enrich History ---
            raw_history_ids = history_map.get(user_id, [])
            enriched_history = []

            for iid in raw_history_ids:
                # Retrieve title/desc if available, else placeholders
                meta = meta_lookup.get(iid, {'title': 'Unknown Item', 'description': ''})
                enriched_history.append({
                    'item_id': iid,
                    'title': meta['title'],
                    'description': meta['description']
                })

            # --- B. Enrich Candidates ---
            enriched_candidates = []
            for _, row in group.iterrows():
                iid = row['item_id']
                meta = meta_lookup.get(iid, {'title': 'Unknown Item', 'description': ''})

                enriched_candidates.append({
                    'item_id': iid,
                    'title': meta['title'],
                    'description': meta['description'],
                    'rank': row['rank'],
                    'score': row['score']
                })

            # --- C. Construct Final Object ---
            user_obj = {
                'user_id': user_id,
                'history': enriched_history,  # The context (past purchases)
                'candidates': enriched_candidates  # The target items to rerank
            }

            # Write as one JSON object per line (JSONL)
            out_f.write(json.dumps(user_obj) + '\n')

    print("Done! Data preparation complete.")


if __name__ == "__main__":
    main()