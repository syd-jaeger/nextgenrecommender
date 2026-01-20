import torch
import streamlit as st
from recbole.quick_start import load_data_and_model


class ModelManager:
    def __init__(self, model_path):
        self.model_path = model_path
        # These will be loaded lazily
        self.config = None
        self.model = None
        self.dataset = None

    @st.cache_resource
    def load(_self):
        """
        Loads the model and dataset. Cached by Streamlit.
        """
        print(f"Loading model from {_self.model_path}...")
        config, model, dataset, _, _, _ = load_data_and_model(
            model_file=_self.model_path
        )
        # Switch to eval mode for inference
        model.eval()
        return config, model, dataset

    def predict(self, external_user_id, top_k=10):
        config, model, dataset = self.load()

        # 1. Map External ID (String) -> Internal ID (Int)
        try:
            user_id_int = dataset.token2id(dataset.uid_field, str(external_user_id))
        except ValueError:
            return None  # User not found

        # 2. Create tensor for inference
        # We score against ALL items
        user_tensor = torch.tensor([user_id_int]).to(config['device']).repeat(dataset.item_num)
        all_item_indices = torch.arange(dataset.item_num).to(config['device'])

        # 3. Predict
        with torch.no_grad():
            scores = model.predict(interaction={
                dataset.uid_field: user_tensor,
                dataset.iid_field: all_item_indices
            })

        # 4. Get Top-K
        _, topk_indices = torch.topk(scores, top_k)

        # 5. Map Internal IDs (Int) -> External IDs (String ASINs)
        recommended_ids = dataset.id2token(dataset.iid_field, topk_indices.cpu())

        return recommended_ids