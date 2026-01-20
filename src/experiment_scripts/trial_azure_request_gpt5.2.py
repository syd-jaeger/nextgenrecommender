import json
import csv
import re
import random
import time
from tqdm import tqdm
from openai import OpenAI

from src.config import RERANKED_DATA_PATH, PROMPTING_DATA_PATH

# --- CONFIGURATION ---
INPUT_FILE = PROMPTING_DATA_PATH / 'sports_outdoors_prompting-data_non_zero_recall_users.jsonl'
OUTPUT_FILE = RERANKED_DATA_PATH / 'sports_outdoors_reranked_azure_gpt5.2_sample_1.csv'
LOG_FILE = RERANKED_DATA_PATH / 'azure_usage_log_gpt5.2_sample_1.csv'
SAMPLE_SIZE = 1
RANDOM_SEED = 42

# --- AZURE SETTINGS ---
AZURE_ENDPOINT = "https://gpt-sweden-gotzian-reuber.openai.azure.com/openai/v1/"  # Hier deine echte Endpoint URL eintragen
AZURE_API_KEY = "eefe02fe836d4eb185d59c6865a2a912"  # Hier deinen Key eintragen
API_VERSION = "2025-04-01-preview"
DEPLOYMENT_NAME = "gpt-5.2-bachelorthesis"

client = OpenAI(
    base_url=f"{AZURE_ENDPOINT}",
    api_key=AZURE_API_KEY,
)

SYSTEM_PROMPT = """
You are an expert Recommender System. Your task is to re-rank a list of candidate items for a specific user based on their purchase history.

CRITICAL INSTRUCTIONS:
1. IGNORE the original order of the candidate items.
2. Rank items purely based on how relevant they are to the user's history.
3. Output ONLY a raw list of Item IDs, separated by spaces.
4. NO JSON, NO MARKDOWN, NO EXPLANATIONS. Just the IDs.
""".strip()


def construct_user_message(user_data):
    history_text = ""
    for item in user_data['history']:
        desc = (item['description'][:200] + '...') if len(item['description']) > 200 else item['description']
        history_text += f"- Title: {item['title']}\n  Description: {desc}\n"

    candidates_text = ""
    for item in user_data['candidates']:
        desc = (item['description'][:200] + '...') if len(item['description']) > 200 else item['description']
        candidates_text += f"Item ID: {item['item_id']}\nTitle: {item['title']}\nDescription: {desc}\n\n"

    prompt = f"""
USER HISTORY (Items bought in the past):
{history_text}

CANDIDATE ITEMS (Please rank these):
{candidates_text}

Response format: ["ITEM_ID_1", "ITEM_ID_2", ...]
"""
    return prompt


def query_azure_gpt(user_message):
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_completion_tokens=500,
            seed=42
        )
        content = response.choices[0].message.content
        usage_stats = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        return content, usage_stats
    except Exception as e:
        print(f"Azure API Error: {e}")
        return None, None  # Wichtig: Muss zwei Werte zurückgeben, damit Unpacking nicht fehlschlägt


# --- UPDATED PARSER: HANDLES SPACES/COMMAS/NEWLINES ---
def parse_llm_response(response_text, original_candidate_ids):
    if not response_text: return []

    # 1. Split string by any whitespace (space, tab, newline) or commas
    # This regex r'[,\s]+' handles "ID1 ID2", "ID1, ID2", or "ID1\nID2" equally well.
    raw_tokens = re.split(r'[,\s]+', response_text.strip())

    # 2. Filter: Keep only tokens that are actual Candidate IDs
    # This automatically removes any accidental text like "Here is the list:"
    valid_ids = [token for token in raw_tokens if token in original_candidate_ids]

    # 3. Handle Missing Items (Same fallback logic as before)
    # If the LLM forgot some items, append them at the end to maintain list length
    missing_ids = [uid for uid in original_candidate_ids if uid not in valid_ids]

    # 4. Deduplicate while preserving order (in case LLM repeated an ID)
    seen = set()
    final_list = []
    for x in valid_ids + missing_ids:
        if x not in seen:
            final_list.append(x)
            seen.add(x)

    return final_list


def main():
    print(f"Loading data from {INPUT_FILE}...")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find input file at {INPUT_FILE}")
        return

    total_users = len(all_lines)
    print(f"Total users available: {total_users}")

    # Random Sampling
    random.seed(RANDOM_SEED)
    if SAMPLE_SIZE < total_users:
        print(f"Sampling {SAMPLE_SIZE} random users...")
        sampled_lines = random.sample(all_lines, SAMPLE_SIZE)
    else:
        sampled_lines = all_lines

    print(f"Starting processing with Azure OpenAI ({DEPLOYMENT_NAME})...")

    # --- HAUPTÄNDERUNG: Dateien EINMAL öffnen und offen lassen ---
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csv_file, \
            open(LOG_FILE, 'w', newline='', encoding='utf-8') as log_file:

        writer = csv.writer(csv_file)
        writer.writerow(['user_id', 'item_id', 'rank'])

        log_writer = csv.writer(log_file)
        log_writer.writerow(['user_id', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'timestamp'])

        for line in tqdm(sampled_lines, desc="Reranking"):
            try:
                user_data = json.loads(line)
                user_id = user_data['user_id']
                candidate_ids = [c['item_id'] for c in user_data['candidates']]

                # Prompt bauen
                prompt = construct_user_message(user_data)

                # --- KORREKTUR 1: Tuple unpacking (response_text UND usage) ---
                response_text, usage = query_azure_gpt(prompt)

                if response_text and usage:
                    # Logging
                    log_writer.writerow([
                        user_id,
                        usage['prompt_tokens'],
                        usage['completion_tokens'],
                        usage['total_tokens'],
                        time.strftime("%Y-%m-%d %H:%M:%S")
                    ])
                    # Sofort auf Festplatte schreiben (hilfreich bei Abstürzen)
                    log_file.flush()

                    # Reranking und Schreiben
                    reranked_ids = parse_llm_response(response_text, candidate_ids)

                    for rank, item_id in enumerate(reranked_ids, start=1):
                        writer.writerow([user_id, item_id, rank])

                    # Auch CSV flushen
                    csv_file.flush()

            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line for User ID {user_id}")
                continue
            except Exception as e:
                print(f"Unexpected error for user {user_id}: {e}")
                continue

    # Die Dateien werden durch das 'with' automatisch hier geschlossen
    print(f"Done! Results: {OUTPUT_FILE}")
    print(f"Usage Logs: {LOG_FILE}")


if __name__ == "__main__":
    main()