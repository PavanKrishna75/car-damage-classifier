from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.dataset import CarDamageDataset


def load_model(model_path: str, num_classes: int):
    checkpoint = torch.load(model_path, map_location="cpu")
    classes = checkpoint["classes"]

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes


def evaluate(
    csv_path: str = "data/train.csv",
    images_root: str = "data",
    model_path: str = "models/best_model.pth",
    batch_size: int = 32,
) -> None:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    ds = CarDamageDataset(csv_path, images_root, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model, classes = load_model(model_path, num_classes=len(ds.classes))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    print("Classes:", classes)
    print()
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()
