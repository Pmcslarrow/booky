# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# # # # # # # # # # # # # # # # # #
#
# USER TOWER
#
#   - user_idx
#
# # # # # # # # # # # # # # # # # #


class UserTower(nn.Module):
    def __init__(self, n_users, embedding_dim, user_embedding_dim=128):
        super().__init__()

        self.user_idx_embedding = nn.Embedding(n_users, user_embedding_dim)

        self.user_mlp = nn.Sequential(
            nn.Linear(user_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, user_idx):
        user_emb = self.user_idx_embedding(user_idx)
        return self.user_mlp(user_emb)


# # # # # # # # # # # # # # # # # #
#
# ITEM TOWER
#
#   - book_idx
#   - book_rank_scaled
#
# # # # # # # # # # # # # # # # # #


class ItemTower(nn.Module):
    def __init__(self, n_books, embedding_dim, book_title_emb_dim, book_embedding_dim=128):
        super().__init__()

        self.book_idx_embedding = nn.Embedding(n_books, book_embedding_dim)

        self.item_mlp = nn.Sequential(
            nn.Linear(book_embedding_dim + 1 + book_title_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, book_idx, book_rank_scaled, book_title_emb):
        book_idx_emb = self.book_idx_embedding(book_idx)

        x = torch.cat([
            book_idx_emb, 
            book_rank_scaled.unsqueeze(1),
            book_title_emb
        ], dim=1)

        return self.item_mlp(x)


# # # # # # # # # # # # # # # # # #
#
# TWO TOWERS
#
# # # # # # # # # # # # # # # # # #


class TwoTowers(nn.Module):
    def __init__(self, user_tower: UserTower, item_tower: ItemTower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, 
                user_idx, 
                book_idx, 
                book_rank_scaled,
                book_title_emb
                ):
        user_emb = self.user_tower(user_idx)
        item_emb = self.item_tower(book_idx, book_rank_scaled, book_title_emb)

        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        return user_emb, item_emb

