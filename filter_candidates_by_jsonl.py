import json
import csv
from pathlib import Path

from src.config import RETRIEVED_DATA_PATH, RERANKED_DATA_PATH, TEMP_DATA_PATH


def load_user_ids_from_jsonl(jsonl_path: Path) -> set:
    user_ids = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Versuche gängige Schlüsselvarianten
            if "user_id" in obj:
                user_ids.add(str(obj["user_id"]))
            elif "User_ID" in obj:
                user_ids.add(str(obj["User_ID"]))
            elif "user" in obj and isinstance(obj["user"], dict):
                # Falls verschachtelt
                uid = obj["user"].get("id") or obj["user"].get("user_id") or obj["user"].get("User_ID")
                if uid is not None:
                    user_ids.add(str(uid))
    return user_ids

def load_user_ids_from_csv(csv_path: Path) -> set:
    user_ids = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"Warnung: Keine Header in {csv_path} gefunden.")
            return user_ids

        # Spalte user_id identifizieren
        preferred = ["user_id"]
        key = next((p for p in preferred if p in reader.fieldnames), None)

        if key is None:
            print(f"Warnung: Spalte 'user_id' nicht in {csv_path} gefunden. Header: {reader.fieldnames}")
            return user_ids

        for row in reader:
            uid = row.get(key)
            if uid is None:
                continue
            uid_str = str(uid).strip()
            if uid_str != "":
                user_ids.add(uid_str)
    print(f"Geladene user_id Einträge aus {csv_path}: {len(user_ids)}")
    return user_ids

def filter_csv_by_user_ids(csv_path: Path, user_ids: set, out_path: Path) -> None:
    # Stelle sicher, dass der Ausgabeordner existiert
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with csv_path.open("r", encoding="utf-8", newline="") as infile, \
         out_path.open("w", encoding="utf-8", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        if not fieldnames or "user_id" not in fieldnames:
            print(f"Warnung: Erwartete Spalte 'user_id' fehlt in {csv_path}. Gefundene Header: {fieldnames}")
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            uid = row.get("user_id")
            uid_str = str(uid).strip() if uid is not None else ""
            if uid_str and uid_str in user_ids:
                writer.writerow(row)
                written += 1
    print(f"Gefilterte Zeilen geschrieben: {written}")

def main():
    base = Path(".")
    reranked_csv_path = RERANKED_DATA_PATH / "movielens_reranked_gpt-4o_azure_batch-v02.csv"
    csv_path = RETRIEVED_DATA_PATH / "movielens-32m/reduced_candidates_non_zero_recall.csv"
    out_path = TEMP_DATA_PATH / "reduced_candidates_non_zero_recall_filtered.csv"

    user_ids = load_user_ids_from_csv(reranked_csv_path)
    if not user_ids:
        print("Keine user_id Einträge in der Reranked-CSV gefunden. Es wird eine leere gefilterte Datei erzeugt.")
    filter_csv_by_user_ids(csv_path, user_ids, out_path)
    print(f"Gefilterte Datei gespeichert unter: {out_path}")

if __name__ == "__main__":
    main()
