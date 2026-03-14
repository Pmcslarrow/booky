import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.two_towers.UserTower import UserTower
from ml.src.utils.dataset import BookRecommenderDataset
from src.utils.config import Config


def calculate_recall_at_k(
        dataset: BookRecommenderDataset, 
        two_towers: UserTower, 
        test_loader: DataLoader,
        epoch: int, 
        config: Config, 
        k: int
    ):
    """
    Gathers the current item embeddings, 
    calculates similarity between each user and the items. 

    Calculates and returns recall@k metric of the recommendations 
    made to the user.
    """

    # --- Getting all item embeddings --- 
    entire_dataset = DataLoader(dataset, batch_size=1, shuffle=False)
    all_item_embeddings = []
    for batch in entire_dataset:
        _, item_embedding = two_towers(batch)
        all_item_embeddings.append(item_embedding)
    all_item_embeddings = torch.cat(all_item_embeddings, dim=0)

    total_recall = 0.0
    num_users = 0

    user_to_books = dataset.data.groupby('User-ID')['Book-Title'].apply(set).to_dict()

    # --- Calculating recall@k --- 
    for idx, batch in enumerate(test_loader):
        user_embedding, _ = two_towers(batch)
        similarity_scores = user_embedding @ all_item_embeddings.T  # [batch_size, num_items]

        top_scores, top_indices = torch.topk(similarity_scores, k=k, dim=1)

        for user_id, items, scores in zip(batch['User-ID'], top_indices, top_scores):
            user_rows = dataset.data[dataset.data['User-ID'] == user_id.item()]

            # Recommended books (Book-Title IDs)
            recommended_book_ids_set = set([dataset.data.iloc[idx.item()]['Book-Title'] for idx in items])
            actual_book_ids_set = set(user_rows['Book-Title'].tolist())

            hits = len(recommended_book_ids_set & actual_book_ids_set)  # intersection
            recall_at_k = hits / len(actual_book_ids_set)

            total_recall += recall_at_k
            num_users += 1

    average_recall_at_k = total_recall / num_users
    print(f"Epoch {epoch}/{config.EPOCHS}, Average Recall@{k}: {average_recall_at_k:.4f}\n")
    return average_recall_at_k
