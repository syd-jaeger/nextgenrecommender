import os
import json
import torch
import numpy as np
import pandas as pd
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

from src import config
from src.config import MODEL_PATH

# === CONFIGURATION ===
NUM_SAMPLES = 100  # How many users to process
SEED = 42  # For reproducibility


# =====================

def run_sampled_inference(model_file, dataset_name):
    print(f"--- Loading Model from {model_file} ---")

    # 1. Load Pre-trained Model & Data
    # config_obj contains the settings used during training
    config_obj, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_file
    )

    # Set device
    model.eval()
    device = config_obj['device']
    model = model.to(device)
    print(f"Model loaded. Processing on device: {device}")

    # --- ID MAPPING & METADATA LOADING ---
    # Amazon datasets often use 'asin' as the item_id.
    # RecBole standardizes this to the configured 'ITEM_ID_FIELD'.
    item_file_path = os.path.join(config_obj['data_path'], f"{dataset_name}.item")
    eid_to_title = {}

    if os.path.exists(item_file_path):
        print(f"Loading metadata from {item_file_path}...")
        # Use on_bad_lines='skip' because Amazon descriptions often contain messy characters
        df_items = pd.read_csv(item_file_path, sep='\t', dtype=str, encoding='utf-8', on_bad_lines='skip')

        # Clean column names (RecBole often uses "field:token")
        df_items.columns = [c.split(':')[0] for c in df_items.columns]

        # Check for title columns (Amazon often has 'title', 'brand', or 'description')
        if 'title' in df_items.columns and 'item_id' in df_items.columns:
            eid_to_title = dict(zip(df_items['item_id'], df_items['title']))
        else:
            print("WARNING: 'title' column not found in metadata. Using ASIN/IDs as titles.")
    else:
        print(f"WARNING: Metadata file not found at {item_file_path}. Titles will be IDs.")

    # --- SAMPLING LOGIC (The Key Change) ---
    print(f"Sampling {NUM_SAMPLES} random users...")

    # Get all unique users who exist in the dataset
    # We use the dataset's total user count to define the range
    # Note: User ID 0 is usually padding in RecBole, so we sample from 1 to num_users
    all_user_ids = np.arange(1, dataset.user_num)

    # Randomly select users
    np.random.seed(SEED)
    sampled_user_ids = np.random.choice(all_user_ids, size=NUM_SAMPLES, replace=False)

    # Convert to Tensor for filtering and processing
    # Keep on CPU for masking, move specific batches to GPU later
    sampled_user_tensor = torch.tensor(sampled_user_ids, dtype=torch.long)

    # --- OPTIMIZED HISTORY LOOKUP ---
    print("Filtering interaction history (Memory Optimization)...")

    # Instead of converting the WHOLE dataset to DataFrame, we filter first.
    # dataset.inter_feat is a huge Dictionary of Tensors
    uid_tensor = dataset.inter_feat[dataset.uid_field]  # All User IDs in interaction history

    # Find indices where the user is in our sampled list
    # torch.isin is available in newer PyTorch versions.
    # If strictly CPU is needed for RAM reasons, use numpy's isin
    mask = torch.isin(uid_tensor, sampled_user_tensor)

    # Filter the features
    subset_uids = uid_tensor[mask].numpy()
    subset_iids = dataset.inter_feat[dataset.iid_field][mask].numpy()
    subset_times = dataset.inter_feat[config_obj['TIME_FIELD']][mask].numpy()

    # Create a small, efficient DataFrame for just these 100 users
    df_inter = pd.DataFrame({
        dataset.uid_field: subset_uids,
        dataset.iid_field: subset_iids,
        config_obj['TIME_FIELD']: subset_times
    })

    print(f"Filtered history contains {len(df_inter)} interactions for {NUM_SAMPLES} users.")

    # --- CANDIDATE GENERATION ---

    # Ensure k is an Integer
    raw_k = config_obj['topk']
    k = max(raw_k) if isinstance(raw_k, (list, tuple)) else int(raw_k)
    print(f"Retrieving Top-{k} candidates.")

    results = []

    # We can process all 100 sampled users in one single batch
    # (unless k is massive or GPU memory is tiny, 100 users is very small)
    batch_users = sampled_user_tensor

    print("Running model inference...")
    with torch.no_grad():
        # Pass CPU tensor 'batch_users', device arg handles GPU move
        _, topk_iids = full_sort_topk(batch_users, model, test_data, k=k, device=device)

        # Move results back to CPU
        topk_iids_cpu = topk_iids.cpu().numpy()
        batch_users_cpu = batch_users.numpy()

        for idx, internal_uid in enumerate(batch_users_cpu):

            # 1. Get History (from our optimized DataFrame)
            user_inter = df_inter[df_inter[dataset.uid_field] == internal_uid]

            # Sort by timestamp
            if config_obj['TIME_FIELD'] in user_inter.columns:
                user_inter = user_inter.sort_values(by=config_obj['TIME_FIELD'])

            full_history_iids = user_inter[dataset.iid_field].values

            # Need at least 2 items (1 for history, 1 for Ground Truth)
            if len(full_history_iids) < 2:
                continue

            # 2. Split History / Ground Truth (Exact same logic as previous script)
            # Last item is GT (Test)
            ground_truth_iid = full_history_iids[-1]

            # Exclude last 2 (Test & Valid) -> Take previous 10
            history_iids = full_history_iids[:-2][-10:]

            if len(history_iids) == 0:
                continue

            # 3. Map IDs to Titles
            def get_title(iid):
                try:
                    eid = dataset.id2token(dataset.iid_field, iid)
                    title = eid_to_title.get(str(eid), str(eid))
                    # Basic cleanup for very long Amazon titles
                    return title[:100] + "..." if len(title) > 100 else title
                except:
                    return str(iid)

            history_titles = [get_title(i) for i in history_iids]
            candidate_titles = [get_title(i) for i in topk_iids_cpu[idx]]
            gt_title = get_title(ground_truth_iid)

            ext_user_id = dataset.id2token(dataset.uid_field, internal_uid)

            results.append({
                "user_id": str(ext_user_id),
                "history": history_titles,
                "candidates": candidate_titles,
                "ground_truth": gt_title,
                "dataset": dataset_name,
                "model": config_obj['model']
            })

    # --- EXPORT ---
    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{config_obj['model']}_{dataset_name}_sample{NUM_SAMPLES}_candidates.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Exported {len(results)} records to {output_file}")


if __name__ == "__main__":
    # --- UPDATE THIS SECTION ---
    # 1. Dataset Name (e.g., 'amazon-books', 'amazon-beauty')
    dataset_name = "amazon_sports_outdoor"

    # 2. Model File Name
    model_file_name = "LightGCN-Nov-20-2025_15-34-50.pth"

    # Construct path
    model_path = os.path.join(MODEL_PATH, dataset_name, model_file_name)

    if os.path.exists(model_path):
        run_sampled_inference(model_path, dataset_name)
    else:
        print(f"File not found: {model_path}")
        print("Please ensure the dataset_name and folder structure match your saved model.")