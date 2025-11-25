import pandas as pd
import faiss
import torch
import pickle
from src.model.two_towers import UserTower

def load_model_and_artifacts():
    """
    Returns:
        model -- UserTower with trained weights
        encoders -- Hashmap encoding of features
        index -- Vector index for items
        dataset -- Original dataset for reference
    """
    with open("data/encoders.pkl", "rb") as f:
        encoder_data = pickle.load(f)

    encoders = encoder_data['encoders']
    num_users = len(encoders['User-ID']) + 1
    num_ages = len(encoders['User-Age']) + 1

    EMBEDDING_SIZE = 128
    model = UserTower(num_users, num_ages, EMBEDDING_SIZE)
    sd = torch.load("models/user_tower.pth", map_location="cpu")
    model.load_state_dict(sd)

    index = faiss.read_index("data/book_items.index")
    dataset = pd.read_pickle("data/dataset_metadata.pkl")

    return model, encoders, index, dataset
