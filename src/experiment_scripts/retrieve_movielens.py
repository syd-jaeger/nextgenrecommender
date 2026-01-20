import os
import json
import torch
import numpy as np
import pandas as pd
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model
from src.config import MODEL_PATH


def run_retrieval_inference(model_file, dataset_name, top_k):
    """
    1. Loads a Pre-trained RecBole Model (.pth).
    2. Generates Top-K Candidates for Test Users.
    3. Exports Data (History + Candidates) to JSONL for LLM.
    """
    print(f"--- Loading Model from {model_file} ---")

    # 1. Load Pre-trained Model & Data
    # This automatically restores config, model weights, and the dataset schema
    config_obj, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_file
    )

    # Ensure model is in eval mode
    model.eval()

    # RecBole often moves model to GPU automatically if config['device'] says so,
    # but let's ensure we use the config's device setting.
    device = config_obj['device']
    model = model.to(device)

    print(f"Model loaded. processing on device: {device}")

    # --- ID MAPPING & METADATA LOADING ---

    # Load Item Metadata to get Titles
    # We use the data_path recovered from the saved config
    item_file_path = os.path.join(config_obj['data_path'], f"{dataset_name}.item")
    eid_to_title = {}

    if os.path.exists(item_file_path):
        # Read file (usually tab separated)
        df_items = pd.read_csv(item_file_path, sep='\t', dtype=str)

        # Clean columns: split returns a list, take [0]
        df_items.columns = [c.split(':')[0] for c in df_items.columns]

        # Create Map: ItemID (str) -> Title (str)
        if 'title' in df_items.columns:
            # Ensure we use the correct external ID column (usually 'item_id')
            eid_to_title = dict(zip(df_items['item_id'], df_items['title']))
        else:
            print("WARNING: 'title' column not found. Using IDs.")
    else:
        print(f"ERROR: Item metadata file not found at {item_file_path}.")
        return

    # --- PRE-PROCESS INTERACTIONS FOR HISTORY LOOKUP ---
    # Convert RecBole Interaction object to Pandas DataFrame for easier lookup logic
    print("Converting interaction data to DataFrame for history lookup...")

    # dataset.inter_feat contains ALL interactions (Train + Valid + Test)
    df_inter = pd.DataFrame({
        dataset.uid_field: dataset.inter_feat[dataset.uid_field].numpy(),
        dataset.iid_field: dataset.inter_feat[dataset.iid_field].numpy(),
        config_obj['TIME_FIELD']: dataset.inter_feat[config_obj['TIME_FIELD']].numpy()
    })

    # --- CANDIDATE GENERATION ---
    k = top_k
    results = []

    # FIX 1: Get unique users as a Numpy array (Keep on CPU)
    test_user_ids = np.unique(test_data.dataset.inter_feat[dataset.uid_field].numpy())

    # Batch Processing
    batch_size = 1024
    num_batches = (len(test_user_ids) + batch_size - 1) // batch_size

    print(f"Generating candidates for {len(test_user_ids)} users in {num_batches} batches...")

    with torch.no_grad():
        for i in range(num_batches):
            # FIX 2: Slice the Numpy array directly. Do NOT move to .to(device) here.
            batch_users = test_user_ids[i * batch_size: (i + 1) * batch_size]

            # Pass CPU numpy array to function. The 'device' arg handles the GPU logic internally.
            _, topk_iids = full_sort_topk(batch_users, model, test_data, k=k, device=device)

            # topk_iids returns as a Tensor (likely on GPU), so we move THAT to CPU
            topk_iids_cpu = topk_iids.cpu().numpy()

            # Iterate over the batch (batch_users is already numpy)
            for idx, internal_uid in enumerate(batch_users):

                # --- History Lookup ---
                user_inter = df_inter[df_inter[dataset.uid_field] == internal_uid]
                if config_obj['TIME_FIELD'] in user_inter.columns:
                    user_inter = user_inter.sort_values(by=config_obj['TIME_FIELD'])

                full_history_iids = user_inter[dataset.iid_field].values
                if len(full_history_iids) < 2: continue

                # Last item = Ground Truth (Test), Previous 10 = History
                ground_truth_iid = full_history_iids[-1]
                history_iids = full_history_iids[:-2][-10:]

                if len(history_iids) == 0: continue

                # --- Mapping ---
                def get_title(iid):
                    try:
                        eid = dataset.id2token(dataset.iid_field, iid)
                        return eid_to_title.get(str(eid), str(eid))
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
                    "model": config_obj['model']
                })

            if (i + 1) % 10 == 0:
                print(f"Processed batch {i + 1}/{num_batches}")

    # --- EXPORT ---
    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)
    # Extract model name from config for filename
    model_name_str = config_obj['model']
    output_file = os.path.join(output_dir, f"{model_name_str}_{dataset_name}_candidates.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Exported {len(results)} records to {output_file}")


if __name__ == "__main__":
    # Example Usage: Point to your .pth file
    # Ensure the path corresponds to where RecBole saved your model
    model_path = MODEL_PATH / "movielens-1m/LightGCN-Dec-08-2025_21-31-45.pth"

    if os.path.exists(model_path):
        run_retrieval_inference(model_path, "movielens-1m", top_k=20)
    else:
        print("Please set the correct path to your .pth model file in the __main__ block.")