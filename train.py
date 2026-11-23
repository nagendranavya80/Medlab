import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data import CLASS_LABELS, create_dataloaders
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DR classifier, optionally including a not_eye class."
    )
    parser.add_argument("--data_dir", type=Path, help="Folder containing train/ and val/.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Train briefly on generated sample images.")
    parser.add_argument("--no_pretrained", action="store_true", help="Do not download ImageNet weights.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs and batch_size must be positive; image_size must be at least 32.")
    if args.data_dir is None and not args.demo:
        parser.error("--data_dir is required unless --demo is set.")
    if args.data_dir is not None and args.demo:
        parser.error("Use either --data_dir or --demo, not both.")
    return args


def create_demo_dataset(root):
    rng = random.Random(7)
    for split in ("train", "val"):
        class_names = [str(index) for index in range(5)] + ["not_eye"]
        for class_name in class_names:
            class_index = 5 if class_name == "not_eye" else int(class_name)
            directory = root / split / class_name
            directory.mkdir(parents=True, exist_ok=True)
            for image_index in range(4 if split == "train" else 2):
                if class_name == "not_eye":
                    image = Image.new(
                        "RGB",
                        (96, 96),
                        (rng.randrange(256), rng.randrange(256), rng.randrange(256)),
                    )
                    draw = ImageDraw.Draw(image)
                    for _ in range(4):
                        x, y = rng.randrange(10, 75), rng.randrange(10, 75)
                        color = tuple(rng.randrange(256) for _ in range(3))
                        draw.rectangle((x, y, x + 12, y + 12), fill=color)
                else:
                    color = (90 + class_index * 25, 55 + class_index * 15, 45)
                    image = Image.new("RGB", (96, 96), (15, 15, 15))
                    draw = ImageDraw.Draw(image)
                    draw.ellipse((8, 8, 88, 88), fill=color)
                    for _ in range(class_index * 3 + 1):
                        x, y = rng.randrange(20, 76), rng.randrange(20, 76)
                        draw.ellipse((x, y, x + 3, y + 3), fill=(230, 30, 30))
                image.save(directory / f"sample_{image_index}.png")


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions, targets = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(targets, predictions), predictions, targets


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temporary_data = tempfile.TemporaryDirectory(prefix="medlab-demo-") if args.demo else None
    try:
        data_dir = Path(temporary_data.name) if temporary_data else args.data_dir
        if args.demo:
            create_demo_dataset(data_dir)
        train_loader, val_loader, class_names, class_weights = create_dataloaders(
            data_dir, args.batch_size, args.image_size, args.num_workers
        )
        model = build_model(
            num_classes=len(class_names),
            pretrained=not args.no_pretrained and not args.demo,
            freeze_backbone=args.demo,
        ).to(device)
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        best_accuracy = -1.0
        best_predictions, best_targets = [], []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy, _, _ = run_epoch(
                model, train_loader, loss_fn, device, optimizer
            )
            val_loss, val_accuracy, predictions, targets = run_epoch(
                model, val_loader, loss_fn, device
            )
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} "
                f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_predictions, best_targets = predictions, targets
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": class_names,
                        "image_size": args.image_size,
                        "val_accuracy": best_accuracy,
                        "demo_training": args.demo,
                    },
                    args.output_dir / "best_model.pth",
                )

        report = classification_report(
            best_targets,
            best_predictions,
            labels=list(range(len(class_names))),
            target_names=[CLASS_LABELS[name] for name in class_names],
            output_dict=True,
            zero_division=0,
        )
        metadata = {
            "architecture": "resnet50",
            "class_names": class_names,
            "class_labels": CLASS_LABELS,
            "image_size": args.image_size,
            "val_accuracy": best_accuracy,
            "demo_training": args.demo,
            "classification_report": report,
        }
        with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"Saved model and metadata to {args.output_dir.resolve()}")
    finally:
        if temporary_data:
            temporary_data.cleanup()


if __name__ == "__main__":
    main()
