import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models

from src.dataset import CarDamageDataset


def create_dataloaders(
    csv_path: str,
    images_root: str,
    batch_size: int = 32,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader, list]:
    ds = CarDamageDataset(csv_path, images_root, augment=True)
    val_size = int(len(ds) * val_split)
    train_size = len(ds) - val_size

    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, ds.classes


def train(
    csv_path: str = "data/train.csv",
    images_root: str = "data",
    models_dir: str = "models",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, classes = create_dataloaders(
        csv_path, images_root, batch_size
    )

    num_classes = len(classes)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Path(models_dir).mkdir(exist_ok=True)
    best_path = Path(models_dir) / "best_model.pth"

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"| Time: {elapsed:.1f}s"
        )

        # ---- Checkpointing ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                },
                best_path,
            )
            print(f"Saved best model → {best_path} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    train()
