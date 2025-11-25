# app.py
import pickle
import torch
from fastapi import FastAPI, HTTPException
import uvicorn
from pathlib import Path
import torch.nn as nn
import io 
import os

class UserTower(nn.Module):

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
    

class Predictor:
    def __init__(self, model_dir="model"):
        base = Path(model_dir)

        with open(base / "encoders.pkl", "rb") as f:
            encoder_data = pickle.load(f)
        self.encoders = encoder_data["encoders"]

        num_users = len(self.encoders["User-ID"]) + 1
        num_ages = len(self.encoders["User-Age"]) + 1
        embedding_dim = 128

        self.model = UserTower(num_users, num_ages, embedding_dim)
        self.model.load_state_dict(torch.load(base / "user_tower.pth", map_location="cpu"))
        self.model.eval()

    def encode_user(self, user_id, age):
        uid = self.encoders['User-ID'].get(str(user_id), 0)
        ageid = self.encoders['User-Age'].get(str(age), 0)
        return torch.tensor([uid], dtype=torch.long), torch.tensor([ageid], dtype=torch.long)

    def predict(self, user_id: int, age_bucket: str):
        uid, uage = self.encode_user(user_id, age_bucket)
        with torch.no_grad():
            emb = self.model(uid, uage).squeeze().tolist()
        return emb


app = FastAPI()
predictor = Predictor(model_dir="model") 

@app.post("/predict")
def predict(request: dict):  
    try:
        user_id = request.get("user_id")
        age = request.get("age")

        # Buckets for age
        if age < 18:
            age_bucket = "<18"
        elif age <= 25:
            age_bucket = "18-25"
        elif age <= 35:
            age_bucket = "26-35"
        elif age <= 50:
            age_bucket = "36-50"
        else:
            age_bucket = "50+"

        if user_id is None or age is None:
            raise HTTPException(status_code=400, detail="Missing user_id or age")
        embedding = predictor.predict(user_id, age_bucket)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

