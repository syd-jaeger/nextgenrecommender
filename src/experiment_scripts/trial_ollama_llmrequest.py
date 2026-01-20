import json
import requests
import csv
import re
import os
import random
from tqdm import tqdm
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")
# --- CONFIGURATION ---
INPUT_FILE = 'sports_outdoors_prompting-data.jsonl'
# Change filename to keep your experiments organized
OUTPUT_FILE = '../../data/reranked/sports_outdoors_reranked_gemma3_sample_350.csv'

# How many users do you want to test?
SAMPLE_SIZE = 350

# Set a seed so your "random" sample is the same every time you run it (reproducibility)
RANDOM_SEED = 42

# Ollama settings
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3"

SYSTEM_PROMPT = """
You are an expert Recommender System. Your task is to re-rank a list of candidate items for a specific user based on their purchase history.

CRITICAL INSTRUCTIONS:
1. IGNORE the original order of the candidate items. The input list is arbitrary.
2. Rank items purely based on how relevant they are to the user's history.
3. Output ONLY a raw JSON list of Item IDs, ordered from most recommended (1st) to least recommended.
4. Do not output any conversational text or markdown. Just the JSON list.
""".strip()


# ... [Previous helper functions: construct_user_message, query_ollama, parse_llm_response remain exactly the same] ...
# (I am omitting them here to save space, paste them back in from the previous script)

def construct_user_message(user_data):
    # ... (Paste from previous script) ...
    # Quick copy-paste for context:
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


def query_ollama(user_message):
    # ... (Paste from previous script) ...
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "stream": False,
        "temperature": 0.3
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json()['message']['content']
    except Exception as e:
        print(f"API Error: {e}")
        return None


def parse_llm_response(response_text, original_candidate_ids):
    # ... (Paste from previous script) ...
    if not response_text: return []
    try:
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        reranked_ids = json.loads(clean_text)
        valid_ids = [uid for uid in reranked_ids if uid in original_candidate_ids]
        missing_ids = [uid for uid in original_candidate_ids if uid not in valid_ids]
        return valid_ids + missing_ids
    except json.JSONDecodeError:
        found_ids = re.findall(r'[a-zA-Z0-9]{10,}', response_text)
        valid_ids = [uid for uid in found_ids if uid in original_candidate_ids]
        seen = set()
        unique_valid = []
        for x in valid_ids:
            if x not in seen:
                unique_valid.append(x)
                seen.add(x)
        missing_ids = [uid for uid in original_candidate_ids if uid not in seen]
        return unique_valid + missing_ids


def main():
    print(f"Loading data from {INPUT_FILE}...")

    # 1. Read all lines first (RAM usage is usually okay for 120k lines unless descriptions are massive)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    total_users = len(all_lines)
    print(f"Total users available: {total_users}")

    # 2. Random Sampling
    random.seed(RANDOM_SEED)  # Ensure we pick the exact same 1000 users if we run this again

    if SAMPLE_SIZE < total_users:
        print(f"Sampling {SAMPLE_SIZE} random users...")
        sampled_lines = random.sample(all_lines, SAMPLE_SIZE)
    else:
        sampled_lines = all_lines

    # 3. Setup Output CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'item_id', 'rank'])

    print(f"Starting processing. Estimated time: {(SAMPLE_SIZE * 13) / 60:.1f} minutes.")

    # 4. Process only the sampled lines
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for line in tqdm(sampled_lines, desc="Reranking Sample"):
            try:
                user_data = json.loads(line)
                user_id = user_data['user_id']
                candidate_ids = [c['item_id'] for c in user_data['candidates']]

                # Build Prompt & Call LLM
                prompt = construct_user_message(user_data)
                response_text = query_ollama(prompt)
                num_tokens = len(encoding.encode(response_text))
                print(num_tokens)
                break

                # Parse & Write
                reranked_ids = parse_llm_response(response_text, candidate_ids)

                for rank, item_id in enumerate(reranked_ids, start=1):
                    writer.writerow([user_id, item_id, rank])

            except json.JSONDecodeError:
                continue

    print(f"Done! Processed {SAMPLE_SIZE} users. Results in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()