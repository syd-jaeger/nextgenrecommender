import streamlit as st
from src import config
from src.backend.inference import ModelManager
from src.backend.metadata import MetadataLoader

st.set_page_config(page_title="Generator - NextGen RecSys", layout="wide")

st.title("⚡ Recommendation Generator")

# --- Retrieve Configuration ---
# Check if paths are in session state, otherwise use defaults
if 'model_path' not in st.session_state:
    st.warning("No model selected. Using default configuration.")
    MODEL_FILE = config.MODEL_PATH / "amazon_sports_outdoor/LightGCN-Nov-20-2025_15-34-50.pth"
    DATA_FILE = config.PROCESSED_DATA_PATH / 'sports_reviews_10core.jsonl'
    META_FILE = config.PROCESSED_DATA_PATH / 'sports_metadata_10core.jsonl'
    selected_dataset_name = "Amazon Sports (10-core)"
else:
    MODEL_FILE = st.session_state['model_path']
    DATA_FILE = st.session_state['data_path']
    META_FILE = st.session_state['meta_path']
    selected_dataset_name = st.session_state.get('selected_dataset', 'Unknown Dataset')

st.markdown(f"**Model:** {MODEL_FILE.name} | **Dataset:** {selected_dataset_name}")

# Initialize Backend
# We use @st.cache_resource to avoid reloading the model on every interaction if possible,
# but for now we'll keep it simple as per original app, or maybe add caching if it's slow.
# The original app didn't use caching explicitly in the main body, but ModelManager might handle it.
# Let's stick to the original logic for now to ensure parity.

try:
    model_manager = ModelManager(model_path=MODEL_FILE)
    meta_loader = MetadataLoader(jsonl_path=META_FILE)
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("User Simulation")
# Determine default user ID based on dataset
default_user = "AHN6PGMHRJOD42YBCIH6MZNO36KA"
if st.session_state.get('selected_dataset') == "MovieLens 1M":
    default_user = "1"

user_input = st.sidebar.text_input("Enter User ID", value=default_user)
k_items = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

if st.sidebar.button("Get Recommendations"):
    with st.spinner("Computing Embeddings..."):
        # 1. Get Raw IDs from Model
        try:
            rec_ids = model_manager.predict(user_input, top_k=k_items)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            rec_ids = None

        if rec_ids is None:
            st.error(f"User ID '{user_input}' not found in training data or error occurred.")
        else:
            # 2. Hydrate with Metadata (Titles/Images)
            rich_results = meta_loader.get_item_details(rec_ids)

            # 3. Display
            st.success("Recommendations generated!")

            # Display in a grid
            cols = st.columns(5)  # 5 items per row
            for i, item in enumerate(rich_results):
                col = cols[i % 5]
                with col:
                    if item.get('image'):
                        st.image(item['image'], use_container_width=True)
                    else:
                        st.write("🖼️ No Image")

                    st.caption(f"Rank {i + 1}")
                    st.markdown(f"**{item.get('title', 'Unknown Title')}**")
                    st.code(item['id'])

# --- DATA INSPECTION (Optional) ---
with st.expander("Debug / System Info"):
    st.write("Using Model Path:", MODEL_FILE)
    st.write("Using Data Path:", DATA_FILE)
    st.write("Using Metadata Path:", META_FILE)
