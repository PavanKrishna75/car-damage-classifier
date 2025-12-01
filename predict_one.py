import sys
from pathlib import Path

from src.infer import DamageClassifier


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python predict_one.py /path/to/image.jpg")
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        raise SystemExit(1)

    clf = DamageClassifier()
    label, conf = clf.predict_from_path(str(image_path))

    print(f"Prediction: {label} (confidence={conf:.3f})")


if __name__ == "__main__":
    main()
