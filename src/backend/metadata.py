import pandas as pd
import streamlit as st
import json


class MetadataLoader:
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path

    @st.cache_data
    def load_mapping(_self):
        """
        Reads the JSONL and creates a dictionary: ID -> {Title, Image}
        Supports both Amazon (ASIN) and MovieLens (id) formats.
        """
        mapping = {}
        # We use a generator to save memory, but for the lookup dict
        # we eventually need to store it in RAM.
        # If 10-core is small enough, this is fine.
        with open(_self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # Check for various ID fields
                    item_id = item.get('parent_asin') or item.get('asin') or item.get('id')
                    title = item.get('title', 'Unknown Product')
                    images = item.get('images', [])
                    
                    # Prefer 'large' image, fallback to 'thumb'
                    image_url = None
                    if images and isinstance(images[0], dict):
                        image_url = images[0].get('large') or images[0].get('thumb')

                    if item_id:
                        # Normalize to string just in case
                        mapping[str(item_id)] = {
                            'title': title,
                            'image': image_url
                        }
                except:
                    continue
        return mapping

    def get_item_details(self, item_id_list):
        mapping = self.load_mapping()
        results = []
        for item_id in item_id_list:
            # metadata keys are strings
            item_id_str = str(item_id)
            details = mapping.get(item_id_str, {'title': item_id_str, 'image': None})
            results.append({
                'id': item_id_str,
                'title': details['title'],
                'image': details['image']
            })
        return results