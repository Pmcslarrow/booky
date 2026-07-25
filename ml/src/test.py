import torch

from ml.src.utils.metrics import Metrics
from ml.src.utils.setup import Setup

if __name__ == "__main__":
    result = Setup(test=True)

    #
    # Getting all item embeddings
    #
    num_items = len(result.test_loader.dataset)
    embedding_dim = result.config.EMBEDDING_SIZE
    all_item_embeddings = torch.zeros((num_items, embedding_dim))
    idx_to_isbn = []  # index i → ISBN

    result.two_towers.eval()
    with torch.no_grad():
        for i, batch in enumerate(result.test_loader):
            item_embedding = result.two_towers.item_tower.get_embedding(batch)
            all_item_embeddings[i] = item_embedding.squeeze(0)
            idx_to_isbn.append(batch["Book-ISBN"][0])
    print("All item embeddings: ", all_item_embeddings.shape)
    print("ISBN mapping size: ", len(idx_to_isbn))

    #
    # Initializing Metric object
    #
    metric = Metrics(
        result.dataset,
        all_item_embeddings,
        result.two_towers.user_tower,
        result.test_loader,
        idx_to_isbn,
    )
    metric.hit_rate_at_k(k=20)
