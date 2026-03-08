import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd


# User Tower -- User-ID, Age
# Item Tower -- ISBN, Book-Title, Book-Author, Publisher, Year-Of-Publication

class BookRecommenderDataset(Dataset):
    """
    A PyTorch Dataset class for book recommendation tasks.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input data containing user, item, and possibly interaction features.

    Attributes
    ----------
    data : pd.DataFrame
        The processed version of the input dataframe.
    encoders : dict
        A dictionary mapping column names to fitted label encoders.
    scalers : dict
        A dictionary mapping column names to fitted scalers for numerical features.
    """

    def __init__(self, data: pd.DataFrame):
        self.encoders = {} # {'Column name': {'value': idx, ...}, ...}
        self.data = data.copy()
        self.original_data = data.copy()
        self.encode_information()

    def encode_information(self):
        """
        Maps {key: index} pairs and StandardScaler for real valued numbers
        """
        label_encoders = ['User-ID', 'ISBN', 'Book-Author', 'Book-Title', 'Publisher', "User-Age", "Book-Year-Of-Publication"]

        for col in label_encoders:
            unique_vals = self.data[col].astype(str).unique()
            self.encoders[col] = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            self.data[col] = self.data[col].astype(str).map(self.encoders[col]).fillna(0).astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        row_original = self.original_data.iloc[idx]

        return {
            "User-ID": torch.tensor(row["User-ID"], dtype=torch.long),
            "User-Age": torch.tensor(row["User-Age"], dtype=torch.long),
            "Book-ISBN": torch.tensor(row["ISBN"], dtype=torch.long),
            "Book-Title": torch.tensor(row["Book-Title"], dtype=torch.long),
            "Book-Author": torch.tensor(row["Book-Author"], dtype=torch.long),
            "Book-Publisher": torch.tensor(row["Publisher"], dtype=torch.long),
            "Book-Year-Of-Publication": torch.tensor(row["Book-Year-Of-Publication"], dtype=torch.long),

            "Book-Title-Text": row_original['Book-Title'],
            "Book-Author-Text": row_original['Book-Author'],
            "Book-ISBN-Text": row_original['ISBN'],
        }


def get_dataloaders(dataset: Dataset, train_p=0.7):
    train_size = int(train_p * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)
    return train_loader, test_loader

