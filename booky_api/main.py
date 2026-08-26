# main.py

import pickle
import torch
import torch.nn.functional as F
from models import UserTower, ItemTower, TwoTowers

### CONSTANTS ### 

KEY_BOOKS = 'books'
KEY_ARTIFACTS = 'artifacts'
KEY_MODEL = 'model'
KEY_STATE_DICT = 'state'

PATH_BOOKS = 'artifacts/books.pkl'
PATH_ARTIFACTS = 'artifacts/artifacts.pkl'
PATH_MODEL = 'artifacts/model.pth'

# NOTE TO SELF:
#   
# Whenever you find yourself needing to
# hardcode some variable that is related to the 
# training phase, you should plan on including 
# that variable as a part of the state_dict that 
# is being saved so it can be referenced here 
# and adhere better to DRY standards.
#

### FUNCTIONS ### 

def load_pickle_file(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

# Returns a dictionary of variables captured during training, used for inference.
def get_model_variables():
    books = load_pickle_file(PATH_BOOKS)
    artifacts = load_pickle_file(PATH_ARTIFACTS)
    state_dict = torch.load(PATH_MODEL)

    n_users = state_dict["n_users"]
    n_books = state_dict["n_books"]
    embedding_dim = state_dict["embedding_dim"]
    book_title_emb_dim = state_dict["book_title_emb_dim"]

    user_tower = UserTower(n_users, embedding_dim)
    item_tower = ItemTower(n_books, embedding_dim, book_title_emb_dim)
    model = TwoTowers(user_tower, item_tower)
    model.load_state_dict(state_dict["model_state"])

    return {
        KEY_BOOKS: books,
        KEY_ARTIFACTS: artifacts,
        KEY_MODEL: model,
        KEY_STATE_DICT: state_dict
    }

@torch.no_grad()
def recommend_for_user(user_idx, artifacts, state, model, k=100, exclude_seen=True):
    model.eval()
    # Embed every book once
    all_books = torch.arange(state['n_books'])
    all_ranks = state['book_rank_scaled_idx']
    book_vectors = F.normalize(model.item_tower(all_books, all_ranks, state['book_title_emb']), p=2, dim=1)

    # Embed the single user
    user_tensor = torch.tensor([user_idx])
    user_vector = F.normalize(model.user_tower(user_tensor), p=2, dim=1)

    # Cosine similarity of this user against every book
    scores = (user_vector @ book_vectors.T).squeeze(0)

    # Hide books the user has already reviewed
    if exclude_seen:
        seen = list(artifacts['user_pos_books'].get(user_idx, set()))
        scores[seen] = float("-inf")

    top_scores, top_idx = torch.topk(scores, k)
    return top_idx.cpu().numpy(), top_scores.cpu().numpy()


def main():
    variables = get_model_variables()
    artifacts = variables[KEY_ARTIFACTS]
    books = variables[KEY_BOOKS]
    model = variables[KEY_MODEL]
    state = variables[KEY_STATE_DICT]

    user_idx = 1000
    top_book_idxs, top_scores = recommend_for_user(
        user_idx, 
        artifacts,
        state, 
        model, 
        k=100
    ) 

    recommendations = books.loc[top_book_idxs, ["book_id", "book_title", "book_rank"]].copy()
    recommendations["score"] = top_scores
    recommendations.reset_index(drop=True)

    print(recommendations.head())

if __name__ == "__main__":
    main()




