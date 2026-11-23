import numpy as np
import torch
import torch.nn.functional as functional
from lime.lime_image import LimeImageExplainer
from PIL import Image
from skimage.segmentation import mark_boundaries

from src.data import build_transforms


def _model_input(image, image_size, device):
    _, transform = build_transforms(image_size)
    return transform(image).unsqueeze(0).to(device)


def grad_cam(model, image, class_index, image_size=224, device="cpu"):
    target_layer = model.layer4[-1].conv3
    activations = []
    gradients = []
    def capture_activation(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    handles = [target_layer.register_forward_hook(capture_activation)]
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(_model_input(image, image_size, device))
        logits[0, class_index].backward()
        weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations[-1]).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(
            heatmap, size=(image.height, image.width), mode="bilinear", align_corners=False
        )
        heatmap = heatmap[0, 0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.detach().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def lime_explanation(model, image, image_size=224, device="cpu", num_samples=500):
    _, transform = build_transforms(image_size)

    def predict(images):
        batch = torch.stack(
            [transform(Image.fromarray(array.astype(np.uint8))) for array in images]
        ).to(device)
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(batch), dim=1).cpu().numpy()

    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.asarray(image.convert("RGB")),
        predict,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )
    label = explanation.top_labels[0]
    _, mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=8, hide_rest=False
    )
    return mark_boundaries(np.asarray(image.convert("RGB")) / 255.0, mask)


def shap_explanation(model, image, class_index, image_size=224, device="cpu"):
    import shap

    model.eval()
    sample = _model_input(image, image_size, device)
    background = torch.zeros_like(sample)
    explainer = shap.GradientExplainer(model, background)
    values = explainer.shap_values(sample, nsamples=32)
    if isinstance(values, list):
        class_values = values[class_index][0]
    else:
        values = np.asarray(values)
        if values.ndim == 5 and values.shape[-1] == 5:
            class_values = values[0, ..., class_index]
        elif values.ndim == 5 and values.shape[1] == 5:
            class_values = values[0, class_index]
        else:
            raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    attribution = np.abs(class_values).mean(axis=0)
    attribution = (attribution - attribution.min()) / (
        attribution.max() - attribution.min() + 1e-8
    )
    return attribution
