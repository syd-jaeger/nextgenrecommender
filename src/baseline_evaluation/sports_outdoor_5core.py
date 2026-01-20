import random

import torch
import numpy as np
import pandas as pd
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model
from src import config
from src.config import ATOMIC_DATA_PATH

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# Settings
model_path = config.MODEL_PATH / 'amazon_sports_outdoor/LightGCN-sports_outdoors_5core_ver_0.4.pth'
test_file_path = config.PROCESSED_DATA_PATH / 'sports_and_outdoors_global_temporal_split/test.csv'
output_csv_name = config.RETRIEVED_DATA_PATH / "LightGCN_ver0.4"  # File to save for reranking

K_METRIC = 20  # For evaluation (Recall@20)
K_GEN = 30  # For export (Top-30 candidates)
BATCH_SIZE = 4096

# ==========================================
# 2. LOAD RESOURCES
# ==========================================

print("Loading model and data...")

checkpoint = torch.load(model_path)
config = checkpoint['config']
config['data_path'] = ATOMIC_DATA_PATH / 'sports_5core'  # Ensure correct data path
dataset = create_dataset(config)
device = config['device']
model = get_model(config['model'])(config, dataset).to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

train_data, valid_data, test_data = data_preparation(config, dataset)

# Load External Test Data
test_df = pd.read_csv(test_file_path, names=['user_id', 'item_id', 'rating', 'timestamp'])
# Create Ground Truth Dictionary: {external_user_id: {external_item_id1, ...}}
ground_truth_dict = test_df.groupby('user_id')['item_id'].apply(set).to_dict()

# Helper Mappings
external_user_ids = list(ground_truth_dict.keys())
uid_field = dataset.uid_field
iid_field = dataset.iid_field


# ==========================================
# 3. METRIC HELPER FUNCTION
# ==========================================

def calculate_metrics(ranked_items, ground_truth, k=20):
    # Truncate to k
    ranked = ranked_items[:k]
    hits = [1 if item in ground_truth else 0 for item in ranked]

    if not ground_truth:
        return 0.0, 0.0, 0.0

    # Recall
    recall = sum(hits) / len(ground_truth)

    # NDCG
    dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
    idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(ground_truth), k))])
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # MRR
    mrr = 0.0
    for idx, hit in enumerate(hits):
        if hit:
            mrr = 1.0 / (idx + 1)
            break

    return recall, ndcg, mrr


# ==========================================
# 4. GENERATION LOOP
# ==========================================

metrics_agg = {'Recall': [], 'NDCG': [], 'MRR': []}
export_data = []  # List to store rows for the CSV

user_mapping = dataset.field2token_id[uid_field]

print(f"Starting inference for {len(external_user_ids)} users...")

for i in range(0, len(external_user_ids), BATCH_SIZE):
    batch_ext_uids = external_user_ids[i: i + BATCH_SIZE]

    batch_int_uids = []
    valid_indices_in_batch = []

    for idx, ext_uid in enumerate(batch_ext_uids):
        # FIX 1 & 2: Use field2token_id (dictionary) instead of token2id (method)
        if ext_uid in user_mapping:
            batch_int_uids.append(user_mapping[ext_uid])
            valid_indices_in_batch.append(idx)

    if not batch_int_uids:
        continue

    # Convert to Tensor
    user_tensor = torch.tensor(batch_int_uids).to(device)

    # FIX 3: Wrap tensor in a dict. RecBole models expect an 'Interaction' object
    # The key must match the dataset's internal user field name (usually 'user_id')
    interaction_input = {uid_field: user_tensor}

    # FIX 4: Use 'full_sort_predict' to get scores for ALL items
    # Output shape: [batch_size, num_items]
    scores = model.full_sort_predict(interaction_input)
    current_batch_size = len(batch_int_uids)
    scores = scores.view(current_batch_size, -1)
    # FIX 5: Manually select Top-K using PyTorch
    # This replaces the missing 'full_sort_topk' function
    # topk_scores: The confidence scores
    # topk_iids: The internal Item IDs
    topk_scores, topk_iids = torch.topk(scores, k=K_GEN)

    # Move to CPU for processing
    topk_scores = topk_scores.cpu().detach().numpy()
    topk_iids = topk_iids.cpu().detach().numpy()

    # Process results (Remains the same as before)
    for j, valid_idx in enumerate(valid_indices_in_batch):
        original_ext_uid = batch_ext_uids[valid_idx]
        user_scores = topk_scores[j]
        user_internal_iids = topk_iids[j]

        predicted_ext_items = []

        for rank, (score, int_iid) in enumerate(zip(user_scores, user_internal_iids)):
            ext_iid = dataset.id2token(iid_field, int_iid)
            predicted_ext_items.append(ext_iid)
            export_data.append([original_ext_uid, ext_iid, rank + 1, score])

        if original_ext_uid in ground_truth_dict:
            actual_items = ground_truth_dict[original_ext_uid]
            r, n, m = calculate_metrics(predicted_ext_items, actual_items, k=K_METRIC)
            metrics_agg['Recall'].append(r)
            metrics_agg['NDCG'].append(n)
            metrics_agg['MRR'].append(m)

print("--- Running Baselines ---")

# ==========================================
# 1. POPULARITY BASELINE (Corrected)
# ==========================================
print("Calculating Global Popularity...")

# Get the raw tensor of item interactions from the dataset
item_tensor = dataset.inter_feat[dataset.iid_field]

# Use PyTorch's bincount to count frequency of every item ID efficiently
# This replaces the .value_counts() which caused the error
item_counts = torch.bincount(item_tensor)

# Get the indices of the top K items with the highest counts
# top_k_indices contains the INTERNAL item IDs
_, popular_indices_tensor = torch.topk(item_counts, k=K_GEN)
popular_items_int = popular_indices_tensor.numpy()

# Convert these Internal IDs -> External IDs (Strings/ASINs)
# We do this once, because Popularity recommends the SAME list to everyone
popular_items_ext = [dataset.id2token(dataset.iid_field, int_id) for int_id in popular_items_int]

# Evaluate Popularity Baseline
pop_metrics = {'Recall': [], 'NDCG': [], 'MRR': []}

for ext_uid, actual_items in ground_truth_dict.items():
    r, n, m = calculate_metrics(popular_items_ext, actual_items, k=K_METRIC)
    pop_metrics['Recall'].append(r)
    pop_metrics['NDCG'].append(n)
    pop_metrics['MRR'].append(m)

# ==========================================
# 2. RANDOM BASELINE
# ==========================================
print("\nCalculating Random Baseline...")

# Identify valid item IDs (RecBole usually reserves 0 for padding)
# Valid items are from 1 to dataset.item_num - 1
valid_item_ids = list(range(1, dataset.item_num))

rand_metrics = {'Recall': [], 'NDCG': [], 'MRR': []}

# We can reuse the loop over ground truth since we need to predict for every test user
for ext_uid, actual_items in ground_truth_dict.items():
    # Pick K random internal IDs
    random_int_ids = random.sample(valid_item_ids, K_GEN)

    # Convert to External IDs
    random_ext_ids = [dataset.id2token(dataset.iid_field, int_id) for int_id in random_int_ids]

    # Evaluate
    r, n, m = calculate_metrics(random_ext_ids, actual_items, k=K_METRIC)
    rand_metrics['Recall'].append(r)
    rand_metrics['NDCG'].append(n)
    rand_metrics['MRR'].append(m)

# ==========================================
# 3. INTERPRETATION
# ==========================================
# Assuming 'metrics_agg' contains your model's results from the previous step
if 'metrics_agg' in locals():
    model_recall = np.mean(metrics_agg['Recall'])
    pop_recall = np.mean(pop_metrics['Recall'])
    rand_recall = np.mean(rand_metrics['Recall'])
# ==========================================
# 5. FINALIZE & SAVE
# ==========================================

# 1. Print Console Metrics
print("\n--- Model Performance (Base) ---")
print(f"Recall@{K_METRIC}: {np.mean(metrics_agg['Recall']):.4f}")
print(f"NDCG@{K_METRIC}:   {np.mean(metrics_agg['NDCG']):.4f}")
print(f"MRR@{K_METRIC}:    {np.mean(metrics_agg['MRR']):.4f}")

print(f"\n[Baseline] Popularity Recall@{K_METRIC}: {np.mean(pop_metrics['Recall']):.5f}")
print(f"[Baseline] Popularity NDCG@{K_METRIC}:   {np.mean(pop_metrics['NDCG']):.5f}")

print(f"[Baseline] Random Recall@{K_METRIC}:     {np.mean(rand_metrics['Recall']):.5f}")
print(f"[Baseline] Random NDCG@{K_METRIC}:       {np.mean(rand_metrics['NDCG']):.5f}")

print("\n--- Comparative Analysis ---")
if pop_recall > 0:
    print(f"Model vs Popularity: {model_recall / pop_recall:.2f}x better")
if rand_recall > 0:
    print(f"Model vs Random:     {model_recall / rand_recall:.2f}x better")

# 2. Save to CSV
print(f"\nSaving {len(export_data)} predictions to {output_csv_name}...")
df_export = pd.DataFrame(export_data, columns=['user_id', 'item_id', 'rank', 'score'])
df_export.to_csv(output_csv_name, index=False)
print("Done!")