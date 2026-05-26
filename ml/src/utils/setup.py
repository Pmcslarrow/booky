import os
import re
import torch
import pandas as pd
from ml.src.models.two_towers import TwoTowers, UserTower, ItemTower
from ml.src.utils.dataset import BookRecommenderDataset, get_dataloaders
from ml.src.utils.config import Config


def Setup(test=False):
    print("Starting setup.py...")
    config = Config()

    #
    # Setup
    #
    cleaned_df = pd.read_csv("ml/data/cleaned/cleaned_dataset.csv")
    # isbn_counts = cleaned_df.groupby('ISBN').filter(lambda x: len(x) > 10)  # TODO: REMOVE
    # cleaned_df = isbn_counts.reset_index(drop=True) # TODO: REMOVE

    personal_df = pd.read_csv("ml/data/personal/paul_books_subset.csv")
    df = pd.concat([cleaned_df, personal_df], ignore_index=True, sort=False)
    book_recommender_dataset = BookRecommenderDataset(df)
    config.set_encoder_lengths(book_recommender_dataset)

    batch_size = 1 if test else config.BATCH_SIZE
    train_loader, test_loader = get_dataloaders(
        book_recommender_dataset, batch_size=batch_size
    )

    print(config)

    #
    # Starting testing
    #
    list_dir = os.listdir("ml/artifacts/models/batch_training/")
    trained_model_path = None
    if list_dir:

        def get_epoch(filename):
            match = re.search(r"epoch(\d+)", filename)
            return int(match.group(1)) if match else -1

        best_file = max(list_dir, key=get_epoch)
        if get_epoch(best_file) >= 0:
            trained_model_path = f"ml/artifacts/models/batch_training/{best_file}"

    #
    # Model
    #
    two_towers = TwoTowers(
        UserTower(
            num_users=config.NUM_USERS,
            num_ages=config.NUM_AGES,
            embedding_dim=config.EMBEDDING_SIZE,
        ),
        ItemTower(
            num_isbns=config.NUM_ISBN,
            num_authors=config.NUM_AUTHORS,
            num_publishers=config.NUM_PUBLISHERS,
            num_year_of_publications=config.NUM_YEAR_OF_PUBLICATIONS,
            embedding_dim=config.EMBEDDING_SIZE,
        ),
    )

    try:
        if trained_model_path:
            checkpoint = torch.load(trained_model_path, map_location="cpu")
            two_towers.load_state_dict(checkpoint)
            print(f"Resuming from checkpoint: {trained_model_path}")
        else:
            print("No checkpoint found, starting from scratch.")
    except Exception as e:
        print("Error loading model weights: ", e)

    return (book_recommender_dataset, config, two_towers, train_loader, test_loader)
