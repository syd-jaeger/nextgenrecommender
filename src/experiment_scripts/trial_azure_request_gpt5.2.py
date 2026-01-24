import json
import csv
import re
import random
import time
from tqdm import tqdm
from openai import OpenAI

from src.config import RERANKED_DATA_PATH, PROMPTING_DATA_PATH, LOG_DATA_PATH
# --- FEW-SHOT EXAMPLES ---

EXAMPLE_2_USER_INPUT = """
User ID: AFTAKXQD7UHKWAG3LH72WAOTFFTQ
USER HISTORY (Items bought in the past):
- Title: OUTDOOR LIVING SUNTIME Camping Folding Portable Mesh Chair with Removabel Footrest
  Description: 
- Title: Seaknight Blade Nylon Fishing Line 500M/1000M Japanese Material Monofilament Line Sea Fishing 2-35LB
  Description: 
- Title: Flex Coat F2S Wrap Finish Kit 2oz With Syringes
  Description: Ultra V's unsurpassed UV protection and improved chemical stability results in unequaled clarity and brightness, while maintaining the highest durabil...
- Title: Reaction Tackle Braided Fishing Line - Pro Grade Power Performance for Saltwater or Freshwater - Colored Diamond Braid for Extra Visibility
  Description: REACTION TACKLE HIGH QUALITY BRAIDED FISHING LINE Reaction Tackle is located in Wisconsin. Everything is shipped from warehouses located in the USA.  ...
- Title: Rapala Fat Boy Fillet Board - White
  Description: Compact for transporting and storage, the Rapala Fat Boy Fillet Board provides ample room with a wide surface to fillet all species of fish. It featur...

CANDIDATE ITEMS (Please rank these):
Item ID: B00C6OUDX2
Title: KastKing SuperPower Braided Fishing Line - Abrasion Resistant Braided Lines – Incredible Superline – Zero Stretch – Smaller Diameter – A Must-Have!
Description: KastKing SuperPower braid line is a fishing line like no other! Our braided fishing lines are designed for increased casting distance and durability. ...

Item ID: B0BHLWZXGC
Title: Fish Scale: Dr.meter PS01 110lb/50kg Backlit LCD Display Fishing Scale with Built-in Measuring Tape - Electronic Balance Digital Fishing Postal Hanging Hook Scale with 2 AAA Batteries
Description: 

Item ID: B0BHJWRQ4N
Title: Berkley Horizontal Fishing Rod Rack, Black, Stores 6 Rods Safely and Securely, Soft Foam Grip Pads, Corrosion Proof Fishing Pole Holder
Description: Safely store and organize your fishing rods with the Vertical Rod Rack. Great for keeping your rods organized in the garage or on a boat, this compact...

Item ID: B005ADORGK
Title: Power Pro Spectra Fiber Braided Fishing Line
Description: PowerPro Microline Today's anglers are more educated than ever. The fact that many bodies of water are getting clearer is exactly why PowerPro has cre...

Item ID: B0BYFLBC89
Title: Reaction Tackle Braided Fishing Line - Pro Grade Power Performance for Saltwater or Freshwater - Colored Diamond Braid for Extra Visibility
Description: REACTION TACKLE HIGH QUALITY BRAIDED FISHING LINE Reaction Tackle is located in Wisconsin. Everything is shipped from warehouses located in the USA.  ...

Item ID: B00OCJHHDI
Title: Piscifun Fishing Line Spooler, No Line Twist Spooling Station System, Fishing Line Winder Spooler for Spinning Reel, Baitcasting Reel and Trolling Reel
Description: 

Item ID: B08WQV6YPR
Title: Berkley PowerBait Natural Glitter Trout Dough Fishing Bait
Description: PowerBait makes novice anglers good and good anglers great! scientists have spent over 25 years perfecting an irresistible scent and flavor - the excl...

Item ID: B0BPMVXSR6
Title: KastKing Summer and Centron Spinning Reels, 9+1 BB Light Weight, Ultra Smooth Powerful, Size 500 is Perfect for Ice Fishing/Ultralight
Description: 

Item ID: B07HFQSPYF
Title: Boomerang Tool Company SNIP Fishing Line Cutters with Retractable Tether and Stainless Steel Blades that Cut Braid, Mono and Fluoro Lines Clean and Smooth!
Description: 

Item ID: B09NR1V42T
Title: Seaguar Blue Label Fluorocarbon Fishing Line Leader, Incredible Impact and Abrasion Resistance, Fast Sinking, Double Structure for Strength and Softness
Description: 


Response format: ITEM_ID_1 ITEM_ID_2 ITEM_ID_3 ...
""".strip()

EXAMPLE_2_ASSISTANT_OUTPUT = "B09NR1V42T B0BHLWZXGC B0BHJWRQ4N B005ADORGK B0BYFLBC89 B00OCJHHDI B08WQV6YPR B0BPMVXSR6 B07HFQSPYF "

# 2. The Mock Output (The "Correct" Answer)
# Logic: Sleeping Bag & Lantern are relevant to Tent. Tennis Racket is not.
# Notice: We return ALL 3 IDs. This teaches the model to not drop items.
EXAMPLE_ASSISTANT_OUTPUT = "B001 B003 B002"


# --- CONFIGURATION ---
INPUT_FILE = PROMPTING_DATA_PATH / 'sports_outdoors_prompting-data-4-user-test.jsonl'
OUTPUT_FILE = RERANKED_DATA_PATH / 'sports_outdoors_reranked_azure_gpt5.2_sample4_prompt_v08.csv'
LOG_FILE = LOG_DATA_PATH / 'azure_usage_log_gpt5.2_sample_4_prompt_v08.csv'
SAMPLE_SIZE = 4
RANDOM_SEED = 42

# --- AZURE SETTINGS ---
AZURE_ENDPOINT = "https://gpt-sweden-gotzian-reuber.openai.azure.com/openai/v1/"
AZURE_API_KEY = "eefe02fe836d4eb185d59c6865a2a912"
API_VERSION = "2025-04-01-preview"
DEPLOYMENT_NAME = "gpt-5.2-bachelorthesis"

client = OpenAI(
    base_url=f"{AZURE_ENDPOINT}",
    api_key=AZURE_API_KEY,
)

SYSTEM_PROMPT = """
You are an expert Recommender System. Your task is to re-rank a list of candidate items for a specific user based on their purchase history.
IGNORE the original order of the candidate items. Rank items purely based on how relevant they are to the user's history.
Output ONLY a raw list of Item IDs, separated by spaces. NO JSON, NO MARKDOWN, NO EXPLANATIONS. Just the IDs.
Here is the first user´s history:
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
Response format: ITEM_ID_1 ITEM_ID_2 ITEM_ID_3 ...
"""
    return prompt


def query_azure_gpt(user_message):
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": EXAMPLE_2_USER_INPUT},
                {"role": "assistant", "content": EXAMPLE_2_ASSISTANT_OUTPUT},
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