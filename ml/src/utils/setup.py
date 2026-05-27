import os
import re
import pickle
import torch
import pandas as pd
from typing import NamedTuple
from ml.src.models.two_towers import TwoTowers, UserTower, ItemTower
from ml.src.utils.dataset import (
    BookRecommenderDataset,
    get_dataloaders,
    get_finetune_loader,
)
from ml.src.utils.config import Config


class SetupResult(NamedTuple):
    """Result of Setup() containing all datasets, loaders, config, and model."""

    dataset: BookRecommenderDataset  # pretrain dataset (main only)
    finetune_dataset: BookRecommenderDataset  # fine-tune dataset (Paul only)
    config: Config
    two_towers: TwoTowers
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    finetune_loader: torch.utils.data.DataLoader


def Setup(test=False):
    print("Starting setup.py...")
    config = Config()

    #
    # Setup: Load datasets
    #
    cleaned_df = pd.read_csv("ml/data/cleaned/cleaned_dataset.csv")
    # isbn_counts = cleaned_df.groupby('ISBN').filter(lambda x: len(x) > 10)  # TODO: REMOVE
    # cleaned_df = isbn_counts.reset_index(drop=True) # TODO: REMOVE

    personal_df = pd.read_csv("ml/data/personal/paul_books_subset.csv")

    # Fit encoders on the combined dataset so Paul's User-ID and ISBNs get valid slots
    combined_df = pd.concat([cleaned_df, personal_df], ignore_index=True, sort=False)
    combined_dataset = BookRecommenderDataset(combined_df)
    config.set_encoder_lengths(combined_dataset)

    # Create pretrain dataset using only main data but with combined-vocab encoders
    pretrain_dataset = BookRecommenderDataset(
        cleaned_df, encoders=combined_dataset.encoders
    )

    # Create Paul's fine-tune dataset using same encoders
    finetune_dataset = BookRecommenderDataset(
        personal_df, encoders=combined_dataset.encoders
    )

    batch_size = 1 if test else config.BATCH_SIZE
    train_loader, test_loader = get_dataloaders(pretrain_dataset, batch_size=batch_size)
    finetune_loader = get_finetune_loader(finetune_dataset, config.FINETUNE_BATCH_SIZE)

    print(config)

    #
    # Checkpoint resumption
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

            encoder_path = trained_model_path.replace(".pt", "_encoders.pkl")
            if os.path.exists(encoder_path):
                with open(encoder_path, "rb") as f:
                    saved_encoders = pickle.load(f)
                # Apply saved encoders to both datasets
                pretrain_dataset.encoders = saved_encoders
                finetune_dataset.encoders = saved_encoders
                pretrain_dataset._apply_encoders()
                finetune_dataset._apply_encoders()
                print(f"Loaded encoders from: {encoder_path}")
            else:
                print(f"Warning: Encoder file not found at {encoder_path}")
        else:
            print("No checkpoint found, starting from scratch.")
    except Exception as e:
        print("Error loading model weights: ", e)

    return SetupResult(
        dataset=pretrain_dataset,
        finetune_dataset=finetune_dataset,
        config=config,
        two_towers=two_towers,
        train_loader=train_loader,
        test_loader=test_loader,
        finetune_loader=finetune_loader,
    )
