import torch
import torch.nn as nn
import torch.nn.functional as F

# # # # # # # # # # # # # # # # # # 
#
# USER TOWER
# 
#   - User-ID
#   - User-Age
#
# # # # # # # # # # # # # # # # # #

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


# # # # # # # # # # # # # # # # # # 
#
# ITEM TOWER
#
#   - Book-Title
#   - Book-Author
#   - Book-Publisher
#   - Book-Year-Of-Publication
#
# # # # # # # # # # # # # # # # # #

class ItemTower(nn.Module):
    def __init__(self, num_isbn, num_titles, num_authors, num_publishers, num_year_of_publications, embedding_dim):
        super().__init__()

        book_title_embedding_size = 32
        book_author_embedding_size = 32
        book_publisher_embedding_size = 16
        book_year_of_publication_embedding_size = 8

        # Categorical embeddings
        self.book_title_embedding = nn.Embedding(num_titles, book_title_embedding_size, padding_idx=0)
        self.book_author_embedding = nn.Embedding(num_authors, book_author_embedding_size, padding_idx=0)
        self.book_publisher_embedding = nn.Embedding(num_publishers, book_publisher_embedding_size, padding_idx=0)
        self.book_year_of_publication_embedding = nn.Embedding(num_year_of_publications, book_year_of_publication_embedding_size, padding_idx=0)

        linear_in = book_title_embedding_size + book_author_embedding_size + book_publisher_embedding_size + book_year_of_publication_embedding_size

        self.item_mlp = nn.Sequential(
            nn.Linear(linear_in, 512), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, book_title, book_author, book_publisher, book_year_of_publication):
        book_title_emb = self.book_title_embedding(book_title)
        book_author_emb = self.book_author_embedding(book_author)
        book_publisher_emb = self.book_publisher_embedding(book_publisher)
        book_year = self.book_year_of_publication_embedding(book_year_of_publication)

        x = torch.cat([
            book_title_emb,
            book_author_emb,
            book_publisher_emb,
            book_year
        ], dim=1)

        return self.item_mlp(x)

    def get_embedding(self, data):
        return self.forward(
            data['Book-Title'],
            data['Book-Author'],
            data['Book-Publisher'],
            data['Book-Year-Of-Publication'],
        )


class TwoTowers(nn.Module):
    def __init__(self, user_tower: UserTower, item_tower: ItemTower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, data):
        user_emb = self.user_tower.get_embedding(data)
        item_emb = self.item_tower.get_embedding(data)

        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        
        return user_emb, item_emb



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

    def forward(self, data):
        user_emb = self.user_tower.get_embedding(data)
        item_emb = self.item_tower.get_embedding(data)

        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        
        return user_emb, item_emb
