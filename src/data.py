from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASS_LABELS = {
    "0": "No DR",
    "1": "Mild",
    "2": "Moderate",
    "3": "Severe",
    "4": "Proliferative DR",
    "not_eye": "Not a retinal fundus image",
}


def build_transforms(image_size=224):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(data_dir, batch_size=32, image_size=224, num_workers=0):
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected train/ and val/ image folders inside {data_path}."
        )

    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("The train and val folders must contain the same class folders.")
    dr_classes = ["0", "1", "2", "3", "4"]
    if train_dataset.classes not in (dr_classes, dr_classes + ["not_eye"]):
        raise ValueError(
            "Expected class folders 0, 1, 2, 3, and 4, optionally followed by not_eye."
        )
    if not train_dataset.samples or not val_dataset.samples:
        raise ValueError("Both train and val folders must contain images.")
    val_counts = torch.bincount(
        torch.tensor(val_dataset.targets), minlength=len(val_dataset.classes)
    )
    if (val_counts == 0).any():
        missing = [
            val_dataset.classes[index]
            for index, count in enumerate(val_counts.tolist())
            if count == 0
        ]
        raise ValueError(f"Validation data is missing images for class folders: {missing}.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    counts = torch.bincount(
        torch.tensor(train_dataset.targets), minlength=len(train_dataset.classes)
    ).float()
    if (counts == 0).any():
        missing = [
            train_dataset.classes[index]
            for index, count in enumerate(counts.tolist())
            if count == 0
        ]
        raise ValueError(f"Training data is missing images for class folders: {missing}.")
    class_weights = counts.sum() / (len(counts) * counts)
    return train_loader, val_loader, train_dataset.classes, class_weights


def load_rgb_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")
