import json

import tiktoken
from tqdm import tqdm

from src.processing_scripts.preprocess_movielens_temporal_split import INPUT_FILE

encoding = tiktoken.encoding_for_model("gpt-4o")
print(len(encoding.encode("You are an expert Recommender System. Your task is to re-rank a list of candidate items for a specific user based on their purchase history. CRITICAL INSTRUCTIONS: 1. IGNORE the original order of the candidate items. The input list is arbitrary.2. Rank items purely based on how relevant they are to the user's history.3. Output ONLY a raw JSON list of Item IDs, ordered from most recommended (1st) to least recommended.4. Do not output any conversational text or markdown. Just the JSON list.")))

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

# INPUT_FILE = "sports_outdoors_prompting-data.jsonl"
# with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#         all_lines = f.readlines()
# token_lengths = []
# for line in tqdm(all_lines, desc="Reranking Sample"):
#     try:
#         user_data = json.loads(line)
#         user_id = user_data['user_id']
#         candidate_ids = [c['item_id'] for c in user_data['candidates']]
#
#         # Build Prompt & Call LLM
#         prompt = construct_user_message(user_data)
#         num_tokens = len(encoding.encode(prompt))
#         token_lengths.append(num_tokens)
#
#     except json.JSONDecodeError:
#         continue
#
#
# print(f"User {user_id} Prompt Token Count: {num_tokens} tokens for {len(candidate_ids)} candidates.")
# print(f"Average Prompt Token Count: {sum(token_lengths)/len(token_lengths):.2f} tokens over {len(token_lengths)} users.")
