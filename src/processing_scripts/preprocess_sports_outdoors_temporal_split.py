import os
import collections
from tqdm import tqdm
from src import config
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

# --- KONFIGURATION ---

# Pfad zur Eingabe-Datei (Deine spezifische Kategorie)
INPUT_FILE = RAW_DATA_PATH / 'Sports_and_Outdoors-5core.csv'

# Ordner, in dem train.csv, valid.csv und test.csv gespeichert werden
OUTPUT_DIR = PROCESSED_DATA_PATH / 'sports_and_outdoors_global_temporal_split/'

# Zeitstempel Grenzen (01.01.2022)
TEST_TIMESTAMP = 1640991600000


# ---------------------

def make_inters_in_order(inters):
    user2inters = collections.defaultdict(list)
    new_inters = collections.defaultdict(list)

    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append(inter)

    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda x: x[3])

        seen_items = set()
        for inter in user_inters:
            item = inter[1]
            if item in seen_items:
                continue
            seen_items.add(item)
            new_inters[user].append(inter)
    return new_inters


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Lese Datei: {INPUT_FILE}")
    inters = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as file:
        header = next(file)  # Header überspringen
        for line in tqdm(file, desc='Lade Daten'):
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            user_id, item_id, rating = parts[0], parts[1], parts[2]
            timestamp = int(parts[3])
            inters.append((user_id, item_id, rating, timestamp))

    print("Sortiere Daten...")
    ordered_inters = make_inters_in_order(inters)

    # --- SCHRITT 1: Daten vorläufig auf Listen verteilen ---
    print("Verteile Daten auf Splits...")
    temp_train = []
    temp_test = []

    for user in ordered_inters:
        for inter in ordered_inters[user]:
            ts = inter[3]
            if ts >= TEST_TIMESTAMP:
                temp_test.append(inter)
            else:
                temp_train.append(inter)

    # --- SCHRITT 2: Welche User/Items sind im Training bekannt? ---
    print("Analysiere Trainingsdaten...")
    train_users = set()
    train_items = set()

    for inter in temp_train:
        train_users.add(inter[0])  # user_id
        train_items.add(inter[1])  # item_id

    print(f"Bekannte User im Training: {len(train_users)}")
    print(f"Bekannte Items im Training: {len(train_items)}")

    # --- SCHRITT 3: Valid/Test filtern (nur bekannte User/Items behalten) ---
    print("Filtere Valid und Test (Cold-Start entfernen)...")

    final_test = []

    removed_valid = 0
    removed_test = 0

    # Filter Test
    for inter in temp_test:
        if inter[0] in train_users and inter[1] in train_items:
            final_test.append(inter)
        else:
            removed_test += 1

    print(f"Entfernt aus Test:  {removed_test} Interaktionen")

    # --- SCHRITT 4: Schreiben ---
    print("Schreibe Dateien...")


    def write_to_csv(filename, data_list):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8', newline='') as f:
            for inter in data_list:
                f.write(f'{inter[0]},{inter[1]},{inter[2]},{inter[3]}\n')


    write_to_csv('train.csv', temp_train)
    write_to_csv('test.csv', final_test)

    print("\nFertig! Statistik (Final):")
    print(f"Train: {len(temp_train)}")
    print(f"Test:  {len(final_test)}")
