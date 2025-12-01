from src.dataset import CarDamageDataset
from torch.utils.data import DataLoader


def main() -> None:
    ds = CarDamageDataset(
        csv_path="data/train.csv",
        images_root="data",
        augment=False,
    )

    print("Size:", len(ds))
    print("Classes:", ds.classes)
    print(ds.label_distribution())

    loader = DataLoader(ds, batch_size=8, shuffle=True)
    imgs, labels = next(iter(loader))

    print("Batch shape:", imgs.shape)
    print("Labels:", labels)


if __name__ == "__main__":
    main()
