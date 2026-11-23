from pathlib import Path
import json
import random
import tempfile
import zipfile

import streamlit as st
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

from src.data import CLASS_LABELS, build_transforms, create_dataloaders
from src.explainability import grad_cam, lime_explanation, shap_explanation
from src.model import build_model


ARTIFACTS = Path("artifacts")
CHECKPOINT = ARTIFACTS / "best_model.pth"
MIN_MODEL_CONFIDENCE = 0.60


@st.cache_resource
def load_model(checkpoint_path, modified_time):
    del modified_time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(num_classes=len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint, device


def safe_extract_archive(archive, destination, max_files=5000, max_bytes=500 * 1024 * 1024):
    destination = destination.resolve()
    total_bytes = 0
    with zipfile.ZipFile(archive) as zip_file:
        members = zip_file.infolist()
        if len(members) > max_files:
            raise ValueError("The archive contains too many files.")
        for member in members:
            member_path = (destination / member.filename).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise ValueError("The archive contains an unsafe file path.")
            total_bytes += member.file_size
            if total_bytes > max_bytes:
                raise ValueError("The archive is larger than the 500 MB upload limit.")
        zip_file.extractall(destination)


def create_showcase_image(disease_index=0):
    rng = random.Random(100 + disease_index)
    image = Image.new("RGB", (360, 360), (20, 30, 24))
    draw = ImageDraw.Draw(image)
    draw.ellipse((35, 35, 325, 325), fill=(190 - disease_index * 12, 120 - disease_index * 8, 75))
    draw.ellipse((115, 115, 245, 245), fill=(235, 180, 120))
    for _ in range(disease_index * 5 + 2):
        x, y = rng.randrange(55, 305), rng.randrange(55, 305)
        radius = rng.randrange(2, 8)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(190, 35, 25))
    return image


def train_uploaded_dataset(dataset_dir, epochs, batch_size, image_size, progress):
    train_loader, val_loader, class_names, class_weights = create_dataloaders(
        dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=False,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_accuracy = -1.0

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(labels.tolist())
        accuracy = accuracy_score(targets, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": image_size,
                    "val_accuracy": best_accuracy,
                    "demo_training": False,
                },
                CHECKPOINT,
            )
        progress.progress((epoch + 1) / epochs)

    metadata = {
        "architecture": "resnet50",
        "class_names": class_names,
        "class_labels": CLASS_LABELS,
        "image_size": image_size,
        "val_accuracy": best_accuracy,
        "demo_training": False,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return best_accuracy


st.set_page_config(page_title="Diabetic Retinopathy Tool", page_icon="👁️")
st.title("Diabetic Retinopathy Diagnostic Tool")
st.warning(
    "Research and decision-support demo only. It is not a diagnosis and must not be "
    "used to guide patient care."
)

tab_demo, tab_upload, tab_training = st.tabs(
    ["Professor demo", "Image prediction", "Train a real model"]
)

with tab_demo:
    st.subheader("Quick demonstration")
    st.caption(
        "This showcase uses simulated fundus imagery. It demonstrates the interface "
        "and explanation workflow, not clinical accuracy."
    )
    showcase_class = st.select_slider(
        "Simulated DR level",
        options=[0, 1, 2, 3, 4],
        format_func=lambda value: CLASS_LABELS[str(value)],
        value=2,
    )
    showcase_image = create_showcase_image(showcase_class)
    st.image(showcase_image, caption="Synthetic fundus sample", width=360)
    if st.button("Run demonstration prediction"):
        st.success(f"Demo output: {CLASS_LABELS[str(showcase_class)]}")
        st.write(
            "The sample and explanation are synthetic. The deployed model is trained "
            "on generated images and must not be used for medical decisions."
        )

if not CHECKPOINT.is_file():
    with tab_upload:
        st.info("No trained model is available. Use the training tab to upload a dataset.")
    with tab_training:
        st.info("Upload a ZIP containing train/ and val/ folders.")
else:
    model, checkpoint, device = load_model(str(CHECKPOINT), CHECKPOINT.stat().st_mtime)
    if checkpoint.get("demo_training", False):
        st.sidebar.error("DEMO MODEL - not clinically valid")
    else:
        st.sidebar.success("Trained model loaded")

    image_size = int(checkpoint["image_size"])
    _, eval_transform = build_transforms(image_size)
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a retinal fundus image",
            type=("jpg", "jpeg", "png"),
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
            tensor = eval_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
            predicted_index = int(probabilities.argmax())
            class_names = checkpoint["class_names"]
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index])
            st.caption(
                f"Model score: {confidence:.1%}. Scores are not calibrated probabilities "
                "and do not indicate diagnostic accuracy."
            )
            if confidence < MIN_MODEL_CONFIDENCE:
                st.error(
                    "This image cannot be classified reliably. It may not be a retinal "
                    "fundus image or may be unlike the training images."
                )
            elif predicted_class == "not_eye":
                st.error(
                    "This does not appear to be a retinal fundus image. Please upload "
                    "a clear retinal fundus photo."
                )
            else:
                st.subheader(f"Model prediction: {CLASS_LABELS[predicted_class]}")
                st.bar_chart(
                    {
                        CLASS_LABELS[class_name]: float(probabilities[index])
                        for index, class_name in enumerate(class_names)
                    }
                )

            method = st.selectbox("Explanation method", ("Grad-CAM", "LIME", "SHAP"))
            if st.button("Generate explanation"):
                try:
                    if method == "Grad-CAM":
                        heatmap = grad_cam(model, image, predicted_index, image_size, device)
                        colored = Image.fromarray((heatmap * 255).astype("uint8")).resize(image.size)
                        overlay = Image.blend(image.convert("RGB"), colored.convert("RGB"), alpha=0.4)
                        st.image(overlay, caption="Grad-CAM attention overlay", use_container_width=True)
                    elif method == "LIME":
                        explanation = lime_explanation(model, image, image_size, device)
                        st.image(explanation, caption="LIME superpixel explanation", use_container_width=True)
                    else:
                        attribution = shap_explanation(model, image, predicted_index, image_size, device)
                        st.image(attribution, caption="Mean absolute SHAP attribution", use_container_width=True)
                except (ImportError, RuntimeError, ValueError) as error:
                    st.error(f"Could not generate the {method} explanation: {error}")

    st.caption(
        f"Validation accuracy on the training split: {checkpoint['val_accuracy']:.1%}"
    )

with tab_training:
    st.subheader("Train on a retinal image dataset")
    st.info(
        "Training runs in the app process and saves the resulting model only for this "
        "session. Streamlit Cloud restarts may remove local checkpoints."
    )
    dataset_archive = st.file_uploader(
        "Upload dataset ZIP",
        type=("zip",),
        help="The ZIP must contain train/<class>/ and val/<class>/ image folders.",
    )
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=20, value=3)
        batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=8)
    with col2:
        image_size = st.number_input("Image size", min_value=32, max_value=512, value=224, step=32)

    if dataset_archive and st.button("Start training", type="primary"):
        try:
            with tempfile.TemporaryDirectory(prefix="medlab-training-") as temp_dir:
                temp_path = Path(temp_dir)
                archive_path = temp_path / "dataset.zip"
                archive_path.write_bytes(dataset_archive.getvalue())
                safe_extract_archive(archive_path, temp_path)
                progress = st.progress(0)
                accuracy = train_uploaded_dataset(
                    temp_path,
                    int(epochs),
                    int(batch_size),
                    int(image_size),
                    progress,
                )
            st.success(f"Training complete. Best validation accuracy: {accuracy:.1%}")
            st.cache_resource.clear()
            st.rerun()
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Training could not start or complete: {error}")
