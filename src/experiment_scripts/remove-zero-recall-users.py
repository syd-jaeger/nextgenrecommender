import pandas as pd
from pathlib import Path
from src import config  # Falls du deine Config nutzen willst, sonst Pfade unten anpassen

# ==========================================
# 1. SETUP & PFADE
# ==========================================

# Eingabe-Dateien
CANDIDATES_PATH = config.RETRIEVED_DATA_PATH / "movielens-32m/ver_0.4_movielens_candidates.csv"  # Deine aktuelle Candidates CSV
GROUND_TRUTH_PATH = config.PROCESSED_DATA_PATH / 'movielens-32m_global_temporal_split/test.csv'

# Ausgabe-Datei
OUTPUT_PATH = config.RETRIEVED_DATA_PATH / "movielens-32m/reduced_candidates_non_zero_recall.csv"

# ==========================================
# 2. DATEN LADEN
# ==========================================
print("Lade Daten...")

# WICHTIG: dtype=str erzwingen, um Typ-Konflikte (int vs str) zu vermeiden
df_candidates = pd.read_csv(CANDIDATES_PATH, dtype={'user_id': str, 'item_id': str})

# Ground Truth hat oft keine Header, daher Namen manuell setzen
df_ground_truth = pd.read_csv(
    GROUND_TRUTH_PATH,
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    dtype={'user_id': str, 'item_id': str}
)

print(f"Original Candidates: {df_candidates['user_id'].nunique()} User")

# ==========================================
# 3. LOGIK: USER MIT HITS FINDEN
# ==========================================

# Wir machen einen Inner Join auf User und Item.
# Das Ergebnis enthält nur Zeilen, wo der Recommender richtig lag (Hits).
hits_df = pd.merge(df_candidates, df_ground_truth, on=['user_id', 'item_id'])

# Wir holen uns die Liste der eindeutigen User-IDs aus diesen Treffern
valid_users = hits_df['user_id'].unique()

print(f"User mit mind. 1 Hit (Recall > 0): {len(valid_users)}")

# ==========================================
# 4. FILTERN UND SPEICHERN
# ==========================================

if len(valid_users) == 0:
    print("WARNUNG: Keine Hits gefunden! Datei wird nicht gespeichert.")
else:
    # Filtere den ursprünglichen Candidates-Dataframe
    # Wir behalten alle Zeilen (auch die falschen Items) von den Usern, die gut performt haben
    df_filtered = df_candidates[df_candidates['user_id'].isin(valid_users)]

    # Speichern
    print(f"Speichere gefilterte Daten nach: {OUTPUT_PATH}")
    df_filtered.to_csv(OUTPUT_PATH, index=False)

    print("Fertig!")