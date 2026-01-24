import pandas as pd
import json
import random
from src.config import PROMPTING_DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH, RETRIEVED_DATA_PATH

# --- CONFIGURATION ---
# Adjust these paths if necessary
METADATA_PATH = RAW_DATA_PATH / 'meta_Sports_and_Outdoors.jsonl'
TRAIN_PATH = PROCESSED_DATA_PATH / 'sports_and_outdoors_global_temporal_split/train.csv'
CANDIDATES_PATH = RETRIEVED_DATA_PATH / 'sports_outdoors_5core/reduced_candidates_non_zero_recall.csv'

# How many examples to generate?
NUM_EXAMPLES = 3


def load_metadata_lookup(filepath):
    """ Loads title and description for items. """
    print("Loading metadata...")
    lookup = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                item_id = data.get('parent_asin')
                if item_id:
                    lookup[item_id] = {
                        'title': data.get('title', 'Unknown'),
                        'description': " ".join(data.get('description', [])) if isinstance(data.get('description'),
                                                                                           list) else str(
                            data.get('description'))
                    }
            except:
                continue
    return lookup


def get_user_history_sample(filepath):
    """ Loads a sample of user histories. """
    print("Loading history...")
    df = pd.read_csv(filepath, names=['user_id', 'item_id', 'rating', 'timestamp'])
    # Get last 5 items per user
    return df.sort_values('timestamp', ascending=False).groupby('user_id')['item_id'].apply(
        lambda x: x.head(5).tolist()).to_dict()


def main():
    # 1. Load Data
    meta_lookup = load_metadata_lookup(METADATA_PATH)
    history_map = get_user_history_sample(TRAIN_PATH)

    print("Loading candidates...")
    df_cands = pd.read_csv(CANDIDATES_PATH)

    # 2. Pick 3 distinct random users
    unique_users = df_cands['user_id'].unique()
    selected_users = random.sample(list(unique_users), NUM_EXAMPLES)

    print(f"\n--- GENERATING {NUM_EXAMPLES} REAL EXAMPLES ---\n")

    for i, user_id in enumerate(selected_users, 1):
        # A. Build History String
        hist_ids = history_map.get(user_id, [])
        history_text = ""
        for iid in hist_ids:
            m = meta_lookup.get(iid, {'title': 'Unknown', 'description': ''})
            desc = (m['description'][:150] + "...") if len(m['description']) > 150 else m['description']
            history_text += f"- Title: {m['title']}\n  Description: {desc}\n"

        # B. Build Candidates String
        # Get this user's candidates
        user_cands = df_cands[df_cands['user_id'] == user_id]['item_id'].tolist()

        candidates_text = ""
        candidate_ids_list = []

        # Limit to 10 candidates for the few-shot example to save tokens (optional, but recommended)
        # We use a smaller subset for examples so the prompt doesn't get huge.
        small_cands = user_cands[:10]

        for iid in small_cands:
            m = meta_lookup.get(iid, {'title': 'Unknown', 'description': ''})
            desc = (m['description'][:150] + "...") if len(m['description']) > 150 else m['description']
            candidates_text += f"Item ID: {iid}\nTitle: {m['title']}\nDescription: {desc}\n\n"
            candidate_ids_list.append(iid)

        # C. Format the Output Block
        print(f"### EXAMPLE {i} (Copy this into your script) ###")
        print(f'EXAMPLE_{i}_USER_INPUT = """')
        print(f"User ID: {user_id}")
        print(f"USER HISTORY (Items bought in the past):")
        print(f"{history_text}")
        print(f"CANDIDATE ITEMS (Please rank these):")
        print(f"{candidates_text}")
        print(f'Response format: ITEM_ID_1 ITEM_ID_2 ITEM_ID_3 ...')
        print('""".strip()')
        print("\n# TODO: MANUALLY REORDER THESE IDs TO CREATE THE 'PERFECT' ANSWER")
        print(f'EXAMPLE_{i}_ASSISTANT_OUTPUT = "{" ".join(candidate_ids_list)}"')
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()