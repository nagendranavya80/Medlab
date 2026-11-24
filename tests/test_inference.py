import unittest

import torch
from PIL import Image
from torch import nn

from src.inference import predict_images


class FixtureModel(nn.Module):
    def forward(self, batch):
        logits = torch.zeros((len(batch), 6))
        for index, image in enumerate(batch):
            brightness = float(image.mean())
            if brightness < 0.1:
                logits[index, 0] = 10
            elif brightness > 0.9:
                logits[index, 5] = 10
        return logits


class WrongOutputModel(nn.Module):
    def forward(self, batch):
        return torch.zeros((len(batch), 5))


def pixel_transform(image):
    value = image.getpixel((0, 0))[0] / 255
    return torch.full((3, 1, 1), value)


class PredictImagesTest(unittest.TestCase):
    def test_returns_individual_screening_results(self):
        images = [
            ("fundus.png", Image.new("RGB", (1, 1), (0, 0, 0))),
            ("uncertain.png", Image.new("RGB", (1, 1), (128, 128, 128))),
            ("other.png", Image.new("RGB", (1, 1), (255, 255, 255))),
        ]

        results = predict_images(
            FixtureModel(),
            images,
            ["0", "1", "2", "3", "4", "not_eye"],
            pixel_transform,
            torch.device("cpu"),
            min_model_score=0.6,
        )

        self.assertEqual(
            [row["Prediction"] for row in results],
            [
                "No DR",
                "Needs review (low model score)",
                "Not a retinal fundus image",
            ],
        )
        self.assertEqual(
            [row["Image"] for row in results],
            ["fundus.png", "uncertain.png", "other.png"],
        )

    def test_rejects_mismatched_model_output(self):
        with self.assertRaisesRegex(ValueError, "output count"):
            predict_images(
                WrongOutputModel(),
                [("fundus.png", Image.new("RGB", (1, 1)))],
                ["0", "1", "2", "3", "4", "not_eye"],
                pixel_transform,
                torch.device("cpu"),
                min_model_score=0.6,
            )
