import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.src.models.two_towers import ItemTower, TwoTowers, UserTower
from ml.src.utils.config import Config
from ml.src.utils.setup import Setup


class Trainer:
    def __init__(
        self,
        config: Config,
        train_loader: DataLoader,
        test_loader: DataLoader,
        encoders: dict,
        interaction_set: frozenset,
        finetune_loader: DataLoader | None = None,
        finetune_interaction_set: frozenset | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.encoders = encoders
        self.finetune_loader = finetune_loader
        self.finetune_interaction_set = finetune_interaction_set or frozenset()
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

    def _save_checkpoint(
        self,
        epoch: int,
        avg_train_loss: float,
        avg_test_loss: float,
        suffix: str = "",
    ):
        path = f"./{self.config.MODEL_SAVE_PATH}/two_towers_epoch{epoch}_test{avg_test_loss:.2f}_train{avg_train_loss:.2f}{suffix}.pt"
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

    def finetune(self):
        """Fine-tune the pretrained model on Paul's personal data."""
        if self.finetune_loader is None:
            raise ValueError("finetune_loader was not provided to Trainer")

        # Create a new optimizer with lower learning rate for fine-tuning
        finetune_optimizer = torch.optim.Adam(
            self.two_towers.parameters(),
            lr=self.config.FINETUNE_LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Temporarily swap optimizer and interaction_set for fine-tune phase
        orig_optimizer = self.optimizer
        orig_interaction_set = self.interaction_set
        self.optimizer = finetune_optimizer
        self.interaction_set = self.finetune_interaction_set

        print("\nStarting fine-tuning on Paul's data...")
        avg_loss = 0.0
        for epoch in range(1, self.config.FINETUNE_EPOCHS + 1):
            avg_loss = self._run_epoch(self.finetune_loader, train=True)
            print(
                f"  Finetune Epoch {epoch}/{self.config.FINETUNE_EPOCHS} | Loss: {avg_loss:.4f}"
            )

        self._save_checkpoint(
            self.config.FINETUNE_EPOCHS, avg_loss, 0.0, suffix="_finetuned"
        )

        # Restore original state
        self.optimizer = orig_optimizer
        self.interaction_set = orig_interaction_set
        print("Fine-tuning complete.")


if __name__ == "__main__":
    result = Setup()

    #
    # Starting training
    #
    trainer = Trainer(
        result.config,
        result.train_loader,
        result.test_loader,
        result.dataset.encoders,
        result.dataset.interaction_set,
        finetune_loader=result.finetune_loader,
        finetune_interaction_set=result.finetune_dataset.interaction_set,
    )

    # Phase 1: Pretrain on main dataset
    trainer.batch_train()

    # Phase 2: Fine-tune on Paul's data
    trainer.finetune()
