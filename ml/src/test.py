import pandas as pd
import torch
from ml.src.models.two_towers import TwoTowers, UserTower, ItemTower
from ml.src.utils.dataset import BookRecommenderDataset, get_dataloaders
from ml.src.utils.config import Config
from ml.src.utils.metrics import Metrics
from ml.src.utils.setup import Setup

if __name__ == "__main__":
    book_recommender_dataset, config, two_towers, train_loader, test_loader = Setup(test=True)

    num_items = len(test_loader.dataset)
    embedding_dim = config.EMBEDDING_SIZE
    all_item_embeddings = torch.zeros((num_items, embedding_dim))

    two_towers.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            item_embedding = two_towers.item_tower.get_embedding(batch)
            all_item_embeddings[i] = item_embedding.squeeze(0)

    print("All item embeddings: ", all_item_embeddings.shape) 

