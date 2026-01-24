from pathlib import Path
import os
CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent
# 2. Baue alle anderen Pfade von diesem Hauptverzeichnis aus auf.
#    Der "/" Operator von pathlib verbindet Pfade intelligent.
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
ATOMIC_DATA_PATH = PROJECT_ROOT / "data" / "atomic"
TEMP_DATA_PATH = PROJECT_ROOT / "data" / "temp"
RETRIEVED_DATA_PATH = PROJECT_ROOT / "data" / "retrieved"
RERANKED_DATA_PATH = PROJECT_ROOT / "data" / "reranked"
PROMPTING_DATA_PATH = PROJECT_ROOT / "data" / "prompting"
MODEL_PATH = PROJECT_ROOT / "models"
LOG_DATA_PATH = PROJECT_ROOT / "log"


if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Models Dir:   {MODEL_PATH}")