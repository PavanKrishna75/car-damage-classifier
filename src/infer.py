from pathlib import Path
from typing import Tuple, List

import torch
from torchvision import models, transforms
from PIL import Image


class DamageClassifier:
    def __init__(
        self,
        model_path: str = "models/best_model.pth",
        device: str | None = None,
    ) -> None:
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        classes: List[str] = checkpoint["classes"]
        num_classes = len(classes)

        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = model.to(self.device)
        self.classes = classes

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        x = self._prepare(image)
        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            conf, idx = probs.max(dim=1)

        label = self.classes[idx.item()]
        confidence = conf.item()
        return label, confidence

    def predict_from_path(self, image_path: str) -> Tuple[str, float]:
        img = Image.open(image_path).convert("RGB")
        return self.predict(img)
