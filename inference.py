import pandas as pd
import faiss
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os 

class UserTower(nn.Module):

    # User Tower -- User-ID, Age

    def __init__(self, num_users, num_ages, embedding_dim):
        super().__init__()

        user_embedding_dim = 128    
        age_embedding_dim = 16   
        linear_in = user_embedding_dim + age_embedding_dim

        self.user_embedding = nn.Embedding(num_users, user_embedding_dim, padding_idx=0)
        self.user_age_embedding = nn.Embedding(num_ages, age_embedding_dim, padding_idx=0)

        self.user_mlp = nn.Sequential(
            nn.Linear(linear_in, 512), # 2 embeddings (user-id, user-age)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, user_id, age):
        user_emb = self.user_embedding(user_id)
        age_emb = self.user_age_embedding(age)
        x = torch.cat([user_emb, age_emb], dim=1)
        return self.user_mlp(x)

    def get_embedding(self, data):
        return self.forward(data['User-ID'], data['User-Age'])


EMBEDDING_SIZE = 128

def load_model_and_artifacts():

    with open("encoders.pkl", "rb") as f:
        encoder_data = pickle.load(f)

    encoders = encoder_data['encoders']
    num_users = len(encoders['User-ID']) + 1
    num_ages = len(encoders['User-Age']) + 1

    model = UserTower(num_users, num_ages, EMBEDDING_SIZE)
    sd = torch.load("user_tower.pth", map_location="cpu")
    model.load_state_dict(sd)

    index = faiss.read_index("book_items.index")
    dataset = pd.read_pickle("dataset_metadata.pkl")

    return model, encoders, index, dataset


def encode_user(user_id, age, encoders):
    user_id = encoders['User-ID'].get(str(user_id), 0)
    user_age = encoders['User-Age'].get(str(age), 0)
    return torch.tensor([user_id], dtype=torch.long), torch.tensor([user_age], dtype=torch.long)


def recommend_books(model, encoders, index, dataset, user_id, age, top_k=100):
    uid, uage = encode_user(user_id, age, encoders)

    with torch.no_grad():
        user_emb = model(uid, uage)
        user_emb = F.normalize(user_emb, p=2, dim=1)
        user_emb = user_emb.numpy()

    distances, indices = index.search(user_emb, top_k)

    results = []
    seen = set()

    for idx, dist in zip(indices[0], distances[0]):
        row = dataset.iloc[idx]
        title = row['Book-Title']
        author = row['Book-Author']
        isbn = row['ISBN']

        if title in seen or isbn in seen:
            continue

        seen.add(title)
        seen.add(isbn)

        results.append({
            "title": title,
            "author": author,
            "isbn": isbn,
            "score": float(dist)
        })

    return results


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    model, encoders, index, dataset = load_model_and_artifacts()
    print("Loaded everything successfully!")
    recs = recommend_books(model, encoders, index, dataset, "1234567890", "18-25", top_k=1000)
    
    for i in range(100):
        title = recs[i]['title']
        author = recs[i]['author']
        distance = recs[i]['score']
        print(f"Title: {title}, Author: {author}, Score: {distance:.4f}")
