import os
from typing import List, Tuple, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CarDamageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_root: str,
        image_col: str = "image",
        label_col: str = "classes",
        classes: Optional[List[str]] = None,
        augment: bool = False,
    ) -> None:
        self.df = pd.read_csv(csv_path)

        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])

        self.image_col = image_col
        self.label_col = label_col
        self.images_root = images_root

        if classes is None:
            self.classes = sorted(self.df[self.label_col].unique().tolist())
        else:
            self.classes = classes

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.02,
                    ),
                ]
                + base_transforms[1:]
            )
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, rel_path: str) -> str:
        # CSV has things like "image/0.jpeg" but files are in data/images/0.jpeg
        candidates = []

        # 1) images_root + rel_path (e.g., data/image/0.jpeg)
        candidates.append(os.path.join(self.images_root, rel_path))

        # 2) basename under images_root/images (e.g., data/images/0.jpeg)
        basename = os.path.basename(rel_path)
        candidates.append(os.path.join(self.images_root, "images", basename))

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"Could not find image for '{rel_path}' in {candidates}")

    def __getitem__(self, idx: int) -> Tuple:
        row = self.df.iloc[idx]
        rel_path = str(row[self.image_col])
        label_str = str(row[self.label_col])

        img_path = self._resolve_image_path(rel_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = self.class_to_idx[label_str]
        return image, label

    def label_distribution(self) -> pd.Series:
        return self.df[self.label_col].value_counts().sort_index()

