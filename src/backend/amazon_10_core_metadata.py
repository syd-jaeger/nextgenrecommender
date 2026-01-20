import pandas as pd
import streamlit as st
import json


class MetadataLoader:
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path

    @st.cache_data
    def load_mapping(_self):
        """
        Reads the JSONL and creates a dictionary: ASIN -> {Title, Image}
        """
        mapping = {}
        # We use a generator to save memory, but for the lookup dict
        # we eventually need to store it in RAM.
        # If 10-core is small enough, this is fine.
        with open(_self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # Adjust keys based on your actual JSON structure
                    asin = item.get('parent_asin') or item.get('asin')
                    title = item.get('title', 'Unknown Product')
                    # Images in Amazon datasets are often lists of URLs
                    images = item.get('images', [])
                    image_url = images[0]['thumb'] if (images and isinstance(images[0], dict)) else None

                    if asin:
                        mapping[asin] = {
                            'title': title,
                            'image': image_url
                        }
                except:
                    continue
        return mapping

    def get_item_details(self, asin_list):
        mapping = self.load_mapping()
        results = []
        for asin in asin_list:
            details = mapping.get(asin, {'title': asin, 'image': None})
            results.append({
                'id': asin,
                'title': details['title'],
                'image': details['image']
            })
        return results