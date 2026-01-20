import streamlit as st
from src import config

st.set_page_config(
    page_title="NextGen Recommender",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ NextGen Recommender System")
st.markdown("""
Welcome to the NextGen Recommender System! 
This platform allows you to generate personalized recommendations using state-of-the-art models.

### Get Started
Select your configuration below to proceed.
""")

# --- Configuration Section ---
st.header("Configuration")

col1, col2 = st.columns(2)

with col2:
    # Dataset Selection
    available_datasets = ["Amazon Sports (10-core)", "MovieLens 1M"]
    selected_dataset = st.selectbox(
        "Select Dataset",
        available_datasets,
        index=0
    )
    st.session_state['selected_dataset'] = selected_dataset

with col1:
    # Model Selection
    if selected_dataset == "Amazon Sports (10-core)":
        available_models = ["amazon_sports_outdoor/LightGCN-Nov-20-2025_15-34-50.pth"]
    else:  # MovieLens 1M
        available_models = ["movielens-1m/LightGCN-Dec-08-2025_21-31-45.pth"]
        
    selected_model = st.selectbox(
        "Select Model", 
        available_models, 
        index=0
    )
    st.session_state['selected_model'] = selected_model

# Store paths in session state for other pages to access
st.session_state['model_path'] = config.MODEL_PATH / selected_model

if selected_dataset == "Amazon Sports (10-core)":
    st.session_state['data_path'] = config.PROCESSED_DATA_PATH / 'sports_reviews_10core.jsonl'
    st.session_state['meta_path'] = config.PROCESSED_DATA_PATH / 'sports_metadata_10core.jsonl'
else:  # MovieLens 1M
    # Point to the raw ratings for reference, and our new metadata file
    st.session_state['data_path'] = config.PROJECT_ROOT / 'data/raw/movielens-1m/ratings.dat'
    st.session_state['meta_path'] = config.PROCESSED_DATA_PATH / 'movielens_metadata.jsonl'

st.markdown("---")
st.info("👈 Select **Generator** from the sidebar to start generating recommendations.")
