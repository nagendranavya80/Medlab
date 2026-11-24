from collections.abc import Callable, Sequence

import torch
from PIL import Image
from torch import Tensor, nn

from src.labels import CLASS_LABELS


def predict_images(
    model: nn.Module,
    images: Sequence[tuple[str, Image.Image]],
    class_names: Sequence[str],
    transform: Callable[[Image.Image], Tensor],
    device: torch.device,
    min_model_score: float,
) -> list[dict[str, str | float]]:
    if not images:
        return []
    if (
        not class_names
        or len(set(class_names)) != len(class_names)
        or any(class_name not in CLASS_LABELS for class_name in class_names)
    ):
        raise ValueError("The model contains an unsupported class name.")

    batch = torch.stack([transform(image) for _, image in images]).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1).cpu()
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != len(images)
        or probabilities.shape[1] != len(class_names)
    ):
        raise ValueError("The model output count does not match its class names.")

    results = []
    for (filename, _), scores in zip(images, probabilities):
        class_index = int(scores.argmax())
        predicted_class = class_names[class_index]
        score = float(scores[class_index])
        if score < min_model_score:
            prediction = "Needs review (low model score)"
        elif predicted_class == "not_eye":
            prediction = CLASS_LABELS["not_eye"]
        else:
            prediction = CLASS_LABELS[predicted_class]

        results.append(
            {
                "Image": filename,
                "Prediction": prediction,
                "Top model class": CLASS_LABELS[predicted_class],
                "Model score": score,
                "Status": (
                    "Review: low model score"
                    if score < min_model_score
                    else "Not a retinal fundus image"
                    if predicted_class == "not_eye"
                    else "Model prediction"
                ),
            }
        )
    return results
