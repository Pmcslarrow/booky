import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from ml.src.models.two_towers import UserTower
from ml.src.utils.dataset import BookRecommenderDataset
from ml.src.utils.config import Config
from collections import defaultdict


class Metrics:
    def __init__(
        self,
        data: BookRecommenderDataset,
        all_item_embeddings,
        user_tower: UserTower,
        test_loader: DataLoader,
        idx_to_isbn: list,
    ):
        self.data = data
        self.all_item_embeddings = all_item_embeddings
        self.user_tower = user_tower
        self.test_loader = test_loader
        self.idx_to_isbn = idx_to_isbn

    def calculate_recall_at_k(
        dataset: BookRecommenderDataset,
        two_towers: UserTower,
        test_loader: DataLoader,
        epoch: int,
        config: Config,
        k: int,
    ):
        """
        Gathers the current item embeddings,
        calculates similarity between each user and the items.

        Calculates and returns recall@k metric of the recommendations
        made to the user.
        """

        # --- Getting all item embeddings ---
        entire_dataset = DataLoader(dataset, batch_size=1, shuffle=False)

        all_item_embeddings = torch.zeros(len(dataset.shape[0]), config.EMBEDDING_SIZE)

        idx = 0
        for batch in entire_dataset:
            with torch.no_grad():
                embeddings = two_towers(batch)
            batch_size = embeddings.size(0)
            all_item_embeddings[idx : idx + batch_size] = embeddings.cpu()
            idx += batch_size

        total_recall = 0.0
        num_users = 0

        # --- Calculating recall@k ---
        for idx, batch in enumerate(test_loader):
            user_embedding, _ = two_towers(batch)
            similarity_scores = (
                user_embedding @ all_item_embeddings.T
            )  # [batch_size, num_items]

            top_scores, top_indices = torch.topk(similarity_scores, k=k, dim=1)

            for user_id, items, scores in zip(
                batch["User-ID"], top_indices, top_scores
            ):
                user_rows = dataset.data[dataset.data["User-ID"] == user_id.item()]

                # Recommended books (Book-Title IDs)
                recommended_book_ids_set = set(
                    [dataset.data.iloc[idx.item()]["Book-Title"] for idx in items]
                )
                actual_book_ids_set = set(user_rows["Book-Title"].tolist())

                hits = len(
                    recommended_book_ids_set & actual_book_ids_set
                )  # intersection
                recall_at_k = hits / len(actual_book_ids_set)

                total_recall += recall_at_k
                num_users += 1

        average_recall_at_k = total_recall / num_users
        print(
            f"Epoch {epoch}/{config.EPOCHS}, Average Recall@{k}: {average_recall_at_k:.4f}\n"
        )
        return average_recall_at_k

    def hit_rate_at_k(self, k):
        """Calculates HR@K

        Args:
            user_features: List[UserFeatures] : List of raw feature vectors for user tower
            ground_truth: List[ItemID] : Single known relevant item for the associated user
            k: Cut off
        """
        print(f"Starting HR@{k}...")
        hits = 0

        test_mapping = defaultdict(list)
        for row in self.test_loader.dataset:
            user_id = row["User-ID"]
            if isinstance(user_id, torch.Tensor):
                user_id = user_id.item()
            test_mapping[user_id].append(row["Book-ISBN"])

        print(len(self.test_loader))

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if not (i % 5000):
                    print(f"Index: {i} --- Hit@{k}@{i}: {hits / (i + 1)}")
                user_emb = self.user_tower.get_embedding(batch)
                scores = user_emb @ self.all_item_embeddings.T

                top_k_indices = np.argsort(-scores.numpy())[0][:k]
                top_k_isbns = [
                    self.idx_to_isbn[i] for i in top_k_indices
                ]  # Model prediction top isbns

                user_id = batch["User-ID"][0]
                if isinstance(user_id, torch.Tensor):
                    user_id = user_id.item()

                relevant_isbns = set(test_mapping[user_id])  # Set of relevant isbns

                relevant_ids = set(t.item() for t in relevant_isbns)
                top_k_ids = set(t.item() for t in top_k_isbns)

                if relevant_ids & top_k_ids:
                    hits += 1

        hit_at_k = hits / len(self.test_loader)
        print(f"Hit@{k} complete --- {hit_at_k}")
        return hit_at_k
