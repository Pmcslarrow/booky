# main.py

import pickle
import torch
from models import UserTower, ItemTower, TwoTowers

### CONSTANTS ### 

KEY_BOOKS = 'books'
KEY_ARTIFACTS = 'artifacts'
KEY_MODEL = 'model'

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
def get_artifacts():
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
        KEY_MODEL: model
    }

def main():
    artifacts = get_artifacts()

if __name__ == "__main__":
    main()




