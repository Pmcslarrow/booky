import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ml.src.utils.dataset import BookRecommenderDataset, get_dataloaders
from ml.src.utils.config import Config
from ml.src.models.two_towers import UserTower, ItemTower, TwoTowers

class Trainer:
    def __init__(self, config: Config, train_loader: DataLoader, test_loader: DataLoader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.two_towers = TwoTowers(
            UserTower(
                num_users=config.NUM_USERS,
                num_ages=config.NUM_AGES,
                embedding_dim=config.EMBEDDING_SIZE
            ),
            ItemTower(
                num_isbn=config.NUM_ISBN,
                num_titles=config.NUM_TITLES,
                num_authors=config.NUM_AUTHORS,
                num_publishers=config.NUM_PUBLISHERS,
                num_year_of_publications=config.NUM_YEAR_OF_PUBLICATIONS,
                embedding_dim=config.EMBEDDING_SIZE
            )
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.two_towers.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.writer = SummaryWriter('./logs/')

    def _compute_loss(self, user_embedding: torch.Tensor, item_embedding: torch.Tensor) -> torch.Tensor:
        logits = (user_embedding @ item_embedding.T) / self.config.TEMPERATURE
        labels = torch.arange(user_embedding.size(0)).to(self.device)
        return F.cross_entropy(logits, labels)

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.two_towers.train() if train else self.two_towers.eval()
        running_loss = 0.0

        with torch.set_grad_enabled(train):
            for batch in loader:
                if train:
                    self.optimizer.zero_grad()

                user_embedding, item_embedding = self.two_towers(batch)
                loss = self._compute_loss(user_embedding, item_embedding)

                if train:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()

        return running_loss / len(loader)

    def _save_checkpoint(self, epoch: int, avg_train_loss: float, avg_test_loss: float):
        path = f"./{self.config.MODEL_SAVE_PATH}/two_towers_epoch{epoch}_test{avg_test_loss:.2f}_train{avg_train_loss:.2f}.pt"
        torch.save(self.two_towers.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    def batch_train(self):
        best_test_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(1, self.config.EPOCHS + 1):
            avg_train_loss = self._run_epoch(self.train_loader, train=True)
            avg_test_loss = self._run_epoch(self.test_loader, train=False)

            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/test', avg_test_loss, epoch)

            print(f"Epoch {epoch}/{self.config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, avg_train_loss, avg_test_loss)

        self.writer.close()
        print("Training complete.")

if __name__ == "__main__":
    print("Starting train.py...")

    #
    # Setup
    #
    cleaned_df = pd.read_csv('ml/data/cleaned/cleaned_dataset.csv')
    isbn_counts = cleaned_df.groupby('ISBN').filter(lambda x: len(x) > 25) # TO REMOVE
    cleaned_df = isbn_counts.reset_index(drop=True) # TO REMOVE
    
    personal_df = pd.read_csv("ml/data/personal/paul_books_subset.csv")
    df = pd.concat([cleaned_df, personal_df], ignore_index=True, sort=False)
    book_recommender_dataset = BookRecommenderDataset(df)
    train_loader, test_loader = get_dataloaders(book_recommender_dataset)

    config = Config(book_recommender_dataset)  
    print(config)

    #
    # Starting training
    #
    trainer = Trainer(config, train_loader, test_loader)
    trainer.batch_train()
