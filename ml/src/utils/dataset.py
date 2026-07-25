import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# User Tower -- User-ID, Age
# Item Tower -- ISBN, Book-Title, Book-Author, Publisher, Year-Of-Publication


class BookRecommenderDataset(Dataset):
    """A PyTorch Dataset class for book recommendation tasks.

    Args:
        Dataset (pd.DataFrame): The input data containing user, item, and possibly interaction features.

    Attributes:
        data (pd.DataFrame): The processed version of the input dataframe.
        encoders (dict): A dictionary mapping column names to fitted label encoders.
        scalers (dict): A dictionary mapping column names to fitted scalers for numerical features.
    """

    def __init__(self, data: pd.DataFrame, encoders: dict | None = None):
        self.encoders = {}  # {'Column name': {'value': idx, ...}, ...}
        self.data = data.copy()
        self.original_data = data.copy()
        if encoders is not None:
            self.encoders = encoders
            self._apply_encoders()
        else:
            self.encode_information()

    def encode_information(self):
        """
        Maps {key: index} pairs and StandardScaler for real valued numbers
        """
        columns = [
            "User-ID",
            "User-Age",
            "ISBN",
            "Book-Author",
            "Book-Title",
            "Book-Year-Of-Publication",
            "Publisher",
        ]

        for col in columns:
            # Drop NaN values and convert to string, then sort
            col_values = self.data[col].dropna().astype(str)
            unique_vals = sorted(col_values.unique())
            self.encoders[col] = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            self.data[col] = (
                self.data[col].astype(str).map(self.encoders[col]).fillna(0).astype(int)
            )

    def _apply_encoders(self):
        """Apply pre-built encoders without re-fitting."""
        columns = [
            "User-ID",
            "User-Age",
            "ISBN",
            "Book-Author",
            "Book-Title",
            "Book-Year-Of-Publication",
            "Publisher",
        ]
        for col in columns:
            self.data[col] = (
                self.data[col].astype(str).map(self.encoders[col]).fillna(0).astype(int)
            )

    @property
    def interaction_set(self) -> frozenset:
        """All (encoded_user_id, encoded_isbn) pairs in the dataset."""
        return frozenset(zip(self.data["User-ID"].tolist(), self.data["ISBN"].tolist()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        # row_original = self.original_data.iloc[idx]

        return {
            "User-ID": torch.tensor(row["User-ID"], dtype=torch.long),
            "User-Age": torch.tensor(row["User-Age"], dtype=torch.long),
            "Book-ISBN": torch.tensor(row["ISBN"], dtype=torch.long),
            "Book-Title": torch.tensor(row["Book-Title"], dtype=torch.long),
            "Book-Author": torch.tensor(row["Book-Author"], dtype=torch.long),
            "Book-Publisher": torch.tensor(row["Publisher"], dtype=torch.long),
            "Book-Year-Of-Publication": torch.tensor(
                row["Book-Year-Of-Publication"], dtype=torch.long
            ),
            # "Book-Title-Text": row_original['Book-Title'],
            # "Book-Author-Text": row_original['Book-Author'],
            # "Book-ISBN-Text": row_original['ISBN'],
        }


def get_dataloaders(
    dataset: Dataset, batch_size, train_p=0.7
) -> tuple[DataLoader, DataLoader]:
    """Split dataset by user-ID to prevent data leakage.

    All interactions from a held-out user go to test; none to train.

    Args:
        dataset (Dataset): BookRecommenderDataset
        train_p (float, optional): Train split percentage. Defaults to 0.7.

    Returns:
        tuple[DataLoader, DataLoader]: train_loader, test_loader
    """
    user_ids = dataset.data["User-ID"].unique()
    np.random.shuffle(user_ids)

    train_user_count = int(train_p * len(user_ids))
    train_users = set(user_ids[:train_user_count])
    test_users = set(user_ids[train_user_count:])

    train_indices = [
        i for i, row in dataset.data.iterrows() if row["User-ID"] in train_users
    ]
    test_indices = [
        i for i, row in dataset.data.iterrows() if row["User-ID"] in test_users
    ]

    train_data = Subset(dataset, train_indices)
    test_data = Subset(dataset, test_indices)

    print("Train size (# rows): ", len(train_data))
    print("Test size (# rows): ", len(test_data))
    print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_finetune_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Return a DataLoader over all rows for fine-tuning on a small personal dataset.

    No train/test split needed for fine-tuning on a small dataset like Paul's 13 rows.

    Args:
        dataset (Dataset): BookRecommenderDataset
        batch_size (int): Batch size for fine-tuning (typically all rows at once)

    Returns:
        DataLoader: Fine-tune loader
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
