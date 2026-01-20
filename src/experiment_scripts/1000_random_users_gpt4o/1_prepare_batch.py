import json
import random
import os
from tqdm import tqdm
from src.config import PROMPTING_DATA_PATH, RERANKED_DATA_PATH

# --- CONFIG ---
INPUT_FILE = PROMPTING_DATA_PATH / 'sports_outdoors_prompting-data_non_zero_recall_users.jsonl'
BATCH_INPUT_FILE = PROMPTING_DATA_PATH / 'batch_input_gpt-4o-v02.jsonl'
CANDIDATES_LOOKUP_FILE = RERANKED_DATA_PATH / 'batch_candidates_lookup.json'  # Needed later for parsing

SAMPLE_SIZE = 1000
RANDOM_SEED = 42
DEPLOYMENT_NAME = "gpt-4o-jonathan-batch"  # Your Azure Deployment Name

SYSTEM_PROMPT = """
You are an expert Recommender System. Your task is to re-rank a list of candidate items for a specific user based on their purchase history.
CRITICAL INSTRUCTIONS:
1. IGNORE the original order of the candidate items.
2. Rank items purely based on how relevant they are to the user's history.
3. Output ONLY a raw list of Item IDs, separated by spaces.
4. CRITICAL: You must return exactly 20 Item IDs for the 20 most relevant items.
5. NO JSON, NO MARKDOWN, NO EXPLANATIONS. Just the IDs.
""".strip()

def construct_prompt(user_data):
    # (Same prompt logic as before)
    history_text = ""
    for item in user_data['history']:
        desc = (item['description'][:150] + '...') if len(item['description']) > 150 else item['description']
        history_text += f"- Title: {item['title']}\n  Description: {desc}\n"

    candidates_text = ""
    for item in user_data['candidates']:
        desc = (item['description'][:150] + '...') if len(item['description']) > 150 else item['description']
        candidates_text += f"Item ID: {item['item_id']}\nTitle: {item['title']}\nDescription: {desc}\n\n"

    return f"USER HISTORY:\n{history_text}\nCANDIDATES:\n{candidates_text}\nResponse format: ID1 ID2 ID3"


def main():
    print("Reading data...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.seed(RANDOM_SEED)
    sampled_lines = random.sample(lines, SAMPLE_SIZE) if len(lines) > SAMPLE_SIZE else lines

    candidates_lookup = {}  # Store original candidates to validte response later

    print(f"Generating batch file for {len(sampled_lines)} users...")

    with open(BATCH_INPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in tqdm(sampled_lines):
            user = json.loads(line)
            user_id = user['user_id']

            # 1. Save candidates for later validation (Step 3)
            candidates_lookup[user_id] = [c['item_id'] for c in user['candidates']]

            # 2. Create Batch Request Object
            request_obj = {
                "custom_id": user_id,  # We use User ID as the request ID
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": DEPLOYMENT_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": construct_prompt(user)}
                    ],
                    "temperature": 0.2,
                    "max_completion_tokens": 300,
                    "seed": 42
                }
            }
            f_out.write(json.dumps(request_obj) + "\n")

    # Save lookup file
    with open(CANDIDATES_LOOKUP_FILE, 'w', encoding='utf-8') as f:
        json.dump(candidates_lookup, f)

    print(f"Done! Ready to upload: {BATCH_INPUT_FILE}")


if __name__ == "__main__":
    main()