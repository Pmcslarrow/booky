# main.py

import os
import pickle
import torch
import torch.nn.functional as F
import functions_framework
from flask import jsonify
from models import UserTower, ItemTower, TwoTowers

### CONSTANTS ###

KEY_BOOKS = 'books'
KEY_ARTIFACTS = 'artifacts'
KEY_MODEL = 'model'
KEY_STATE_DICT = 'state'

PATH_BOOKS = 'artifacts/books.pkl'
PATH_ARTIFACTS = 'artifacts/artifacts.pkl'
PATH_MODEL = 'artifacts/model.pth'

### FUNCTIONS ### 

def load_pickle_file(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def get_model_variables():
    books = load_pickle_file(PATH_BOOKS)
    artifacts = load_pickle_file(PATH_ARTIFACTS)
    state_dict = torch.load(PATH_MODEL, map_location="cpu")

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
_VARS = get_model_variables()
_ARTIFACTS = _VARS[KEY_ARTIFACTS]
_BOOKS = _VARS[KEY_BOOKS]
_MODEL = _VARS[KEY_MODEL]
_STATE = _VARS[KEY_STATE_DICT]

### INFERENCE ###

@torch.no_grad()
def recommend_for_user(user_idx, artifacts, state, model, k=100, exclude_seen=True):
    all_books = torch.arange(state['n_books'])
    all_ranks = state['book_rank_scaled_idx']
    book_vectors = F.normalize(
        model.item_tower(all_books, all_ranks, state['book_title_emb']), p=2, dim=1
    )

    user_tensor = torch.tensor([user_idx])
    user_vector = F.normalize(model.user_tower(user_tensor), p=2, dim=1)

    scores = (user_vector @ book_vectors.T).squeeze(0)

    if exclude_seen:
        seen = list(artifacts['user_pos_books'].get(user_idx, set()))
        scores[seen] = float("-inf")

    k = min(k, scores.shape[0])
    top_scores, top_idx = torch.topk(scores, k)
    return top_idx.cpu().numpy(), top_scores.cpu().numpy()

### HTTP ENTRY POINT ###

@functions_framework.http
def recommend(request):
    """
    Query params:
      user_idx (int, required)
      k        (int, optional, default 100)
      exclude_seen (bool, optional, default true)
    """
    args = request.args

    user_idx_raw = args.get("user_idx")
    if user_idx_raw is None:
        return jsonify({"error": "missing required param 'user_idx'"}), 400

    try:
        user_idx = int(user_idx_raw)
    except ValueError:
        return jsonify({"error": "'user_idx' must be an integer"}), 400

    if not (0 <= user_idx < _STATE["n_users"]):
        return jsonify({"error": f"'user_idx' out of range [0, {_STATE['n_users']})"}), 400

    k = int(args.get("k", 100))
    exclude_seen = args.get("exclude_seen", "true").lower() != "false"

    try:
        top_book_idxs, top_scores = recommend_for_user(
            user_idx, _ARTIFACTS, _STATE, _MODEL, k=k, exclude_seen=exclude_seen
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    recommendations = _BOOKS.loc[top_book_idxs, ["book_id", "book_title", "book_rank"]].copy()
    recommendations["score"] = top_scores
    recommendations = recommendations.reset_index(drop=True)

    return jsonify({
        "user_idx": user_idx,
        "k": k,
        "recommendations": recommendations.to_dict(orient="records")
    })