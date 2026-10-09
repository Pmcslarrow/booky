# main.py

import io
import os
import pickle

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from google.cloud import storage

from .models import ItemTower, TwoTowers, UserTower

app = Flask(__name__)

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/recommend")

### CONSTANTS ###

KEY_BOOKS = 'books'
KEY_ARTIFACTS = 'artifacts'
KEY_MODEL = 'model'
KEY_STATE_DICT = 'state'

PATH_BOOKS = "books.pkl"
PATH_ARTIFACTS = "artifacts.pkl"
PATH_MODEL = "model.pth"

### FUNCTIONS ### 

def load_torch_file(path: str):
    if path.startswith("gs://"):
        buffer = io.BytesIO(get_blob(path).download_as_bytes())
        return torch.load(buffer, map_location="cpu")
    return torch.load(path, map_location="cpu")

def get_blob(gcs_path):
    bucket_name, blob_path = gcs_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    return client.bucket(bucket_name).blob(blob_path)

def load_pickle_file(path: str):
    if path.startswith("gs://"):
        blob = get_blob(path)
        return pickle.loads(blob.download_as_bytes())
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model_variables():
    base_path = os.environ.get("AIP_STORAGE_URI", "artifacts")
    if not base_path:
        raise ValueError("AIP_STORAGE_URI is missing. Ensure this is running inside Vertex AI.")

    books_file_path = os.path.join(base_path, PATH_BOOKS)
    artifacts_file_path = os.path.join(base_path, PATH_ARTIFACTS)
    model_file_path = os.path.join(base_path, PATH_MODEL)

    print(f"Loading model weights from: {model_file_path}")
    print(f"Loading artifacts from {artifacts_file_path}")
    print(f"Loading books from {books_file_path}")

    books = load_pickle_file(books_file_path)
    artifacts = load_pickle_file(artifacts_file_path)
    state_dict = load_torch_file(model_file_path)

    n_users = state_dict["n_users"]
    n_books = state_dict["n_books"]
    embedding_dim = state_dict["embedding_dim"]
    book_title_emb_dim = state_dict["book_title_emb_dim"]

    user_tower = UserTower(n_users, embedding_dim)
    item_tower = ItemTower(n_books, embedding_dim, book_title_emb_dim)
    model = TwoTowers(user_tower, item_tower)
    model.load_state_dict(state_dict["model_state"])
    model.eval()

    return {
        KEY_BOOKS: books,
        KEY_ARTIFACTS: artifacts,
        KEY_MODEL: model,
        KEY_STATE_DICT: state_dict,
    }

# This runs ONCE per container instance, not per-request.
_VARS = load_model_variables()
_ARTIFACTS = _VARS[KEY_ARTIFACTS]
_BOOKS = _VARS[KEY_BOOKS]
_MODEL = _VARS[KEY_MODEL]
_STATE = _VARS[KEY_STATE_DICT]

with torch.no_grad():
    _BOOK_VECTORS = F.normalize(
        _MODEL.item_tower(
            torch.arange(_STATE["n_books"]),
            _STATE["book_rank_scaled_idx"],
            _STATE["book_title_emb"],
        ),
        p=2, dim=1,
    )


### INFERENCE ###

@torch.no_grad()
def recommend_for_user(user_idx, k=100, exclude_seen=True):
    user_vector = F.normalize(_MODEL.user_tower(torch.tensor([user_idx])), p=2, dim=1)
    scores = (user_vector @ _BOOK_VECTORS.T).squeeze(0)

    if exclude_seen:
        seen = list(_ARTIFACTS["user_pos_books"].get(user_idx, set()))
        if seen:
            scores[seen] = float("-inf")

    k = min(k, scores.shape[0])
    top_scores, top_idx = torch.topk(scores, k)
    return top_idx.cpu().numpy(), top_scores.cpu().numpy()

@app.route(HEALTH_ROUTE, methods=["GET"])
def health():
    return "OK", 200

@app.route(PREDICT_ROUTE, methods=["POST"])
def recommend():
    body = request.get_json(silent=True) or {}
    instances = body.get("instances")
    if not instances:
        return jsonify({"error": "body must contain non-empty 'instances'"}), 400

    predictions = []
    try:
        for inst in instances:
            user_idx = int(inst["userIdx"])
            k = int(inst.get("k", 100))

            if k <= 0:
                return jsonify({"error": "'k' must be positive"}), 400
            if not (0 <= user_idx < _STATE["n_users"]):
                return jsonify({"error": f"'userIdx' out of range [0, {_STATE['n_users']})"}), 400

            idxs, scores = recommend_for_user(user_idx, k=k)
            recs = _BOOKS.iloc[idxs][["book_id", "book_title", "book_rank"]].copy()
            recs["score"] = scores
            predictions.append({
                "userIdx": user_idx,
                "k": k,
                "recommendations": recs.to_dict(orient="records"),
            })
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"bad request: {e}"}), 400
    except (RuntimeError, OSError):
        return jsonify({"error": "internal error"}), 500

    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run()