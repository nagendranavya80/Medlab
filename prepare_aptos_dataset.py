import argparse
import csv
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DR_CLASSES = ("0", "1", "2", "3", "4")


def find_images(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_and_copy(files, class_name, source_dir, output_dir, validation_fraction, rng):
    shuffled = list(files)
    rng.shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)

    for split, split_files in (
        ("train", shuffled[validation_count:]),
        ("val", shuffled[:validation_count]),
    ):
        for image_path in split_files:
            relative_path = image_path.relative_to(source_dir)
            destination = output_dir / split / class_name / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)


def prepare_dataset(
    images_dir,
    labels_csv,
    not_eye_dir,
    output_dir,
    validation_fraction=0.2,
    seed=42,
):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Labels CSV does not exist: {labels_csv}")

    images_by_id = {}
    for image_path in find_images(images_dir):
        if image_path.stem in images_by_id:
            raise ValueError(f"Duplicate image ID found: {image_path.stem}")
        images_by_id[image_path.stem] = image_path

    files_by_class = {class_name: [] for class_name in DR_CLASSES}
    seen_ids = set()
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not {"id_code", "diagnosis"}.issubset(reader.fieldnames or ()):
            raise ValueError("APTOS CSV must contain id_code and diagnosis columns.")
        for row in reader:
            image_id = (row["id_code"] or "").strip()
            try:
                diagnosis = str(int(row["diagnosis"]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid diagnosis for image {image_id!r}."
                ) from error
            if not image_id or diagnosis not in files_by_class:
                raise ValueError(f"Invalid APTOS label row for image {image_id!r}.")
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image ID in labels CSV: {image_id}")
            if image_id not in images_by_id:
                raise FileNotFoundError(f"No image file found for ID {image_id}.")
            seen_ids.add(image_id)
            files_by_class[diagnosis].append(images_by_id[image_id])

    files_by_class["not_eye"] = find_images(not_eye_dir)
    for class_name, files in files_by_class.items():
        if len(files) < 2:
            raise ValueError(
                f"Class {class_name} needs at least two images for train and validation."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for class_name, files in files_by_class.items():
        split_and_copy(
            files,
            class_name,
            not_eye_dir if class_name == "not_eye" else images_dir,
            output_dir,
            validation_fraction,
            rng,
        )

    return {
        split: {
            class_name: sum(
                1
                for path in (output_dir / split / class_name).rglob("*")
                if path.is_file()
            )
            for class_name in files_by_class
        }
        for split in ("train", "val")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare APTOS and non-eye images for the DR app."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--not-eye-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        counts = prepare_dataset(
            args.images_dir,
            args.labels_csv,
            args.not_eye_dir,
            args.output_dir,
            args.validation_fraction,
            args.seed,
        )
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))

    print(f"Prepared dataset at {args.output_dir.resolve()}")
    for split, class_counts in counts.items():
        print(f"{split}: {class_counts}")


if __name__ == "__main__":
    main()
