import json
import csv
import re
from src.config import RERANKED_DATA_PATH

BATCH_OUTPUT_FILE = RERANKED_DATA_PATH / 'batch_output_gpt-4o_v02.jsonl'
CANDIDATES_LOOKUP_FILE = RERANKED_DATA_PATH / 'batch_candidates_lookup.json'
FINAL_CSV = RERANKED_DATA_PATH / 'sports_outdoors_reranked_azure_batch-v02.csv'


def parse_ids(response_text, valid_candidates):
    # Same regex logic as before
    raw_tokens = re.split(r'[,\s]+', response_text.strip())
    valid_ids = [t for t in raw_tokens if t in valid_candidates]

    # Deduplicate
    seen = set()
    unique = []
    for x in valid_ids:
        if x not in seen:
            unique.append(x)
            seen.add(x)

    # Fallback: fill missing
    missing = [c for c in valid_candidates if c not in seen]
    return unique + missing


def main():
    print("Loading lookup data...")
    with open(CANDIDATES_LOOKUP_FILE, 'r') as f:
        candidates_lookup = json.load(f)

    print("Processing batch results...")
    with open(BATCH_OUTPUT_FILE, 'r', encoding='utf-8') as f_in, \
            open(FINAL_CSV, 'w', newline='', encoding='utf-8') as f_out:

        writer = csv.writer(f_out)
        writer.writerow(['user_id', 'item_id', 'rank'])

        for line in f_in:
            data = json.loads(line)

            # The 'custom_id' we sent is the User ID
            user_id = data['custom_id']

            # Extract content from the nested response
            # Azure Batch response structure: response -> body -> choices -> message -> content
            response_content = data['response']['body']['choices'][0]['message']['content']

            # Get valid candidates for this user
            valid_candidates = candidates_lookup.get(user_id, [])

            # Parse
            ranked_ids = parse_ids(response_content, valid_candidates)

            # Write
            for rank, item_id in enumerate(ranked_ids, start=1):
                writer.writerow([user_id, item_id, rank])

    print(f"Finished! Final CSV: {FINAL_CSV}")


if __name__ == "__main__":
    main()