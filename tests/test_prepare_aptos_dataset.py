import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_aptos_dataset import prepare_dataset


class PrepareAptosDatasetTest(unittest.TestCase):
    def test_creates_all_classes_in_stratified_splits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images_dir = root / "aptos" / "train_images"
            not_eye_dir = root / "other_images"
            output_dir = root / "prepared"
            images_dir.mkdir(parents=True)
            not_eye_dir.mkdir()
            labels_csv = root / "train.csv"

            with labels_csv.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=("id_code", "diagnosis"))
                writer.writeheader()
                for grade in range(5):
                    for image_index in range(2):
                        image_id = f"grade{grade}_{image_index}"
                        Image.new("RGB", (4, 4), (grade, image_index, 0)).save(
                            images_dir / f"{image_id}.png"
                        )
                        writer.writerow(
                            {"id_code": image_id, "diagnosis": grade}
                        )

            for image_index in range(4):
                Image.new("RGB", (4, 4), (image_index, 0, 0)).save(
                    not_eye_dir / f"other_{image_index}.png"
                )

            counts = prepare_dataset(
                images_dir,
                labels_csv,
                not_eye_dir,
                output_dir,
                validation_fraction=0.5,
                seed=1,
            )

            self.assertEqual(set(counts["train"]), {"0", "1", "2", "3", "4", "not_eye"})
            for class_name in ("0", "1", "2", "3", "4"):
                self.assertEqual(counts["train"][class_name], 1)
                self.assertEqual(counts["val"][class_name], 1)
            self.assertEqual(counts["train"]["not_eye"], 2)
            self.assertEqual(counts["val"]["not_eye"], 2)

    def test_rejects_missing_non_eye_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images_dir = root / "aptos"
            not_eye_dir = root / "other_images"
            images_dir.mkdir()
            not_eye_dir.mkdir()
            labels_csv = root / "train.csv"
            with labels_csv.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=("id_code", "diagnosis"))
                writer.writeheader()
                for grade in range(5):
                    for image_index in range(2):
                        image_id = f"grade{grade}_{image_index}"
                        Image.new("RGB", (4, 4)).save(
                            images_dir / f"{image_id}.png"
                        )
                        writer.writerow(
                            {"id_code": image_id, "diagnosis": grade}
                        )

            with self.assertRaisesRegex(ValueError, "Class not_eye needs"):
                prepare_dataset(
                    images_dir,
                    labels_csv,
                    not_eye_dir,
                    root / "prepared",
                )
