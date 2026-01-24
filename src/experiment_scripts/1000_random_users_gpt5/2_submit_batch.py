import datetime
import time
from openai import OpenAI
from src.config import RERANKED_DATA_PATH, PROMPTING_DATA_PATH

# --- AZURE CONFIG ---
AZURE_ENDPOINT = "https://gpt-sweden-gotzian-reuber.openai.azure.com/openai/deployments/gpt-5-jonathan-batch/chat/completions?api-version=2025-01-01-preview"
AZURE_API_KEY = "eefe02fe836d4eb185d59c6865a2a912"
API_VERSION = "2025-01-01-preview"

BATCH_INPUT_FILE = PROMPTING_DATA_PATH / 'batch_input_gpt-5-v02.jsonl'
OUTPUT_FILENAME = "batch_output_gpt-5-v02.jsonl"  # Name to save downloaded results

client = OpenAI(
    base_url = "https://gpt-sweden-gotzian-reuber.openai.azure.com/openai/v1/",
    api_key=AZURE_API_KEY,
)


def main():
    # 1. Upload File
    print("Uploading file to Azure...")
    batch_file = client.files.create(
        file=open(BATCH_INPUT_FILE, "rb"),
        purpose="batch"
    )
    file_id = batch_file.id
    print(f"File uploaded. ID: {file_id}")

    # 2. Create Batch Job
    print("Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=file_id,
        endpoint="/chat/completions",
        completion_window="24h"  # Standard window
    )
    batch_id = batch_job.id
    print(f"Batch job started! ID: {batch_id}")
    print("Waiting for completion (this may take a while)...")

    status = "validating"
    while status not in ("completed", "failed", "canceled"):
        time.sleep(60)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

    if batch_response.status == "failed":
        for error in batch_response.errors.data:
            print(f"Error code {error.code} Message {error.message}")

    # 4. Download Results
    print("Job completed! Downloading results...")
    output_file_id = batch_job.output_file_id
    content = client.files.content(output_file_id).content

    save_path = RERANKED_DATA_PATH / OUTPUT_FILENAME
    with open(save_path, 'wb') as f:
        f.write(content)

    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()