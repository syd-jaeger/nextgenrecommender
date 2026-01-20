import polars as pl
import os
os.environ["POLARS_LARGE_INDICES"] = "0"
import shutil
import io
import gc

from src import config

# --- CONFIGURATION ---
INPUT_FILE = config.PROCESSED_DATA_PATH / 'Sports_and_Outdoors_processed.jsonl'
TEMP_FOLDER = config.TEMP_DATA_PATH / 'temp_parquet_parts'  # Folder to store chunk files
OUTPUT_FILE = config.PROCESSED_DATA_PATH / 'sports_and_outdoors_10core.jsonl'
MIN_INTERACTIONS = 10

# Column names
USER_COL = 'reviewerID'
ITEM_COL = 'parent_asin'  # or 'asin'

# How many lines to process at once.
# 500,000 is a safe balance between speed and RAM (approx 500MB - 1GB RAM usage)
CHUNK_SIZE = 50_000

# ---------------------

def chunk_jsonl_to_parquet(input_path, output_folder):
    """
    Reads JSONL in small safe batches and saves to Parquet.
    Includes explicit Garbage Collection to prevent MemoryError.
    """
    print(f"Step 1: Splitting {input_path} into Parquet chunks...")
    print(f"   Chunk size: {CHUNK_SIZE} rows")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    chunk_buffer = []
    chunk_count = 0
    total_lines = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk_buffer.append(line)
                total_lines += 1

                # When buffer hits the limit, process it
                if len(chunk_buffer) >= CHUNK_SIZE:
                    _save_chunk(chunk_buffer, output_folder, chunk_count)
                    chunk_count += 1

                    # CLEAR MEMORY IMMEDIATELY
                    chunk_buffer = []
                    gc.collect()  # Force Python to release RAM now

                    print(f"   Processed {total_lines:,} lines...")

            # Process remaining lines
            if chunk_buffer:
                _save_chunk(chunk_buffer, output_folder, chunk_count)

    except MemoryError:
        print(f"!!! CRITICAL MEMORY ERROR at line {total_lines} !!!")
        print("Try reducing CHUNK_SIZE further (e.g., to 10,000) or setting DROP_HEAVY_TEXT = True")
        return False

    print(f"Conversion complete. Created {chunk_count + 1} parquet parts.")
    return True

def _save_chunk(lines_list, folder, index):
    """Helper to parse a list of JSON strings and save to Parquet"""
    # Join lines into a single string block for Polars to parse from memory
    json_data = io.BytesIO("".join(lines_list).encode('utf-8'))

    try:
        # infer_schema_length=0 forces reading all to find types, safer for chunks
        df = pl.read_ndjson(json_data)

        # Ensure IDs are strings to prevent schema mismatch between chunks
        # (e.g. chunk 1 has "123" (int), chunk 2 has "ABC" (str))
        df = df.with_columns([
            pl.col(USER_COL).cast(pl.String),
            pl.col(ITEM_COL).cast(pl.String)
        ])

        output_path = os.path.join(folder, f"part_{index:04d}.parquet")
        df.write_parquet(output_path)
    except Exception as e:
        print(f"   Warning: Error saving chunk {index}: {e}")


def get_k_core_from_folder(folder_path, k, user_col, item_col):
    """
    Scans a folder of Parquet files and applies the k-core filter.
    """
    print(f"\nStep 2: Lazy Loading Parquet parts from {folder_path}...")

    # scan_parquet can read a directory of files as if it were one file
    lazy_df = pl.scan_parquet(os.path.join(folder_path, "*.parquet"))

    # Check size
    total_rows = lazy_df.select(pl.len()).collect().item()
    print(f"Total rows loaded: {total_rows}")

    # Iterative Filtering
    print(f"Starting {k}-core processing...")
    iteration = 0

    # We maintain the dataframe in memory now (it should fit as Parquet is efficient)
    # If this crashes, we can switch to a pure Lazy approach, but try this first.
    df = lazy_df.collect()

    while True:
        iteration += 1
        start_len = len(df)

        # Filter
        df = df.filter(pl.len().over(user_col) >= k)
        df = df.filter(pl.len().over(item_col) >= k)

        end_len = len(df)
        print(f"Iteration {iteration}: {start_len} -> {end_len} rows")

        if start_len == end_len:
            print("Convergence reached.")
            break

    return df


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. Manual Chunking (Bypasses InputTooLarge error)
    chunk_jsonl_to_parquet(INPUT_FILE, TEMP_FOLDER)

    # 2. Process the chunks
    df_core = get_k_core_from_folder(TEMP_FOLDER, MIN_INTERACTIONS, USER_COL, ITEM_COL)

    # Stats
    n_users = df_core[USER_COL].n_unique()
    n_items = df_core[ITEM_COL].n_unique()

    print(f"\nFinal Stats:")
    print(f"Total Interactions: {len(df_core)}")
    print(f"Unique Users: {n_users}")
    print(f"Unique Items: {n_items}")

    # 3. Save
    print(f"\nStep 3: Saving result to {OUTPUT_FILE}...")
    df_core.write_ndjson(OUTPUT_FILE)

    # Optional: Cleanup temp folder
    # shutil.rmtree(TEMP_FOLDER)
    print("Done.")


if __name__ == "__main__":
    main()