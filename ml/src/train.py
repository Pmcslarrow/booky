import torch
import torch.nn.functional as F
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from ml.src.utils.config import Config
from ml.src.models.two_towers import UserTower, ItemTower, TwoTowers
from ml.src.utils.setup import Setup


class Trainer:
    def __init__(
        self,
        config: Config,
        train_loader: DataLoader,
        test_loader: DataLoader,
        encoders: dict,
        interaction_set: frozenset,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.encoders = encoders
        # self.EarlyStopping = EarlyStopping() TODO: Implement an early stopping class
        # self.Writer = Writer() TODO: Implement a custom writer class

        self.two_towers = TwoTowers(
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
        ).to(self.device)

        self.interaction_set = interaction_set

        self.optimizer = torch.optim.Adam(
            self.two_towers.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    def _compute_loss(
        self,
        user_embedding: torch.Tensor,
        item_embedding: torch.Tensor,
        user_ids: torch.Tensor,
        isbn_ids: torch.Tensor,
    ) -> torch.Tensor:
        B = user_embedding.size(0)
        logits = (user_embedding @ item_embedding.T) / self.config.TEMPERATURE

        # Mask off-diagonal entries that are known (user, isbn) interactions so
        # they are not penalised as false negatives during contrastive learning.
        mask = torch.zeros(B, B, dtype=torch.bool, device=self.device)
        ui = user_ids.tolist()
        ii = isbn_ids.tolist()
        for i in range(B):
            for j in range(B):
                if i != j and (ui[i], ii[j]) in self.interaction_set:
                    mask[i, j] = True
        logits = logits.masked_fill(mask, float("-inf"))

        labels = torch.arange(B, device=self.device)
        return F.cross_entropy(logits, labels)

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.two_towers.train() if train else self.two_towers.eval()
        running_loss = 0.0

        with torch.set_grad_enabled(train):
            for batch in loader:
                if train:
                    self.optimizer.zero_grad()

                user_embedding, item_embedding = self.two_towers(batch)
                loss = self._compute_loss(
                    user_embedding,
                    item_embedding,
                    batch["User-ID"].to(self.device),
                    batch["Book-ISBN"].to(self.device),
                )

                if train:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()

        return running_loss / len(loader)

    def _save_checkpoint(self, epoch: int, avg_train_loss: float, avg_test_loss: float):
        path = f"./{self.config.MODEL_SAVE_PATH}/two_towers_epoch{epoch}_test{avg_test_loss:.2f}_train{avg_train_loss:.2f}.pt"
        torch.save(self.two_towers.state_dict(), path)

        encoder_path = path.replace(".pt", "_encoders.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(self.encoders, f)

        print(f"  Checkpoint saved: {path}")
        print(f"  Encoders saved: {encoder_path}")

    def batch_train(self):
        for epoch in range(1, self.config.EPOCHS + 1):
            avg_train_loss = self._run_epoch(self.train_loader, train=True)
            avg_test_loss = self._run_epoch(self.test_loader, train=False)

            print(
                f"Epoch {epoch}/{self.config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}"
            )

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, avg_train_loss, avg_test_loss)

        print("Training complete.")


if __name__ == "__main__":
    book_recommender_dataset, config, two_towers, train_loader, test_loader = Setup()

    #
    # Starting training
    #
    trainer = Trainer(
        config,
        train_loader,
        test_loader,
        book_recommender_dataset.encoders,
        book_recommender_dataset.interaction_set,
    )
    trainer.batch_train()
