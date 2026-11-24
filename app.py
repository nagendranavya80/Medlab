import csv
import io
import json
import random
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageDraw, UnidentifiedImageError
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

from src.data import CLASS_LABELS, build_transforms, create_dataloaders
from src.explainability import grad_cam, lime_explanation, shap_explanation
from src.inference import predict_images
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


st.set_page_config(
    page_title="RetinaCare | Research workspace",
    page_icon=":material/visibility:",
    layout="wide",
)
st.title("RetinaCare", icon=":material/visibility:")
st.caption("Diabetic retinopathy research workspace")
st.warning(
    "Research and decision-support demo only. It is not a diagnosis and must not be "
    "used to guide patient care."
)

model = checkpoint = device = class_names = None
if CHECKPOINT.is_file():
    model, checkpoint, device = load_model(str(CHECKPOINT), CHECKPOINT.stat().st_mtime)
    class_names = checkpoint["class_names"]
    if checkpoint.get("demo_training", False):
        st.sidebar.error("Synthetic demo model - predictions disabled")
    elif set(class_names) != set(CLASS_LABELS):
        st.sidebar.warning("Model needs retraining")
    else:
        st.sidebar.success("Model available")
else:
    st.sidebar.info("No model checkpoint available")

tab_home, tab_upload, tab_training, tab_demo = st.tabs(
    ["Home", "Image screening", "Train a model", "Example demo"]
)

with tab_home:
    with st.container(border=True):
        intro_column, status_column = st.columns([1.7, 1], vertical_alignment="center")
        with intro_column:
            st.markdown(":blue-badge[RETINAL IMAGE WORKFLOW]")
            st.header(
                "A clear workspace for retinal image review",
                icon=":material/visibility:",
            )
            st.write(
                "Screen JPG and PNG images in batches, review model outputs, and "
                "export a detection list. Train with labeled data to enable screening."
            )
            st.caption("Choose Image screening or Train a model from the tabs above.")

        with status_column:
            st.markdown("#### Model readiness")
            if checkpoint is None:
                st.badge("No model installed", color="gray")
                st.caption("Train a model to enable image screening.")
            elif checkpoint.get("demo_training", False):
                st.badge("Synthetic demo", color="orange")
                st.caption(
                    "Real-image predictions are disabled until a model is trained "
                    "with labeled retinal data."
                )
            elif set(class_names) != set(CLASS_LABELS):
                st.badge("Needs retraining", color="orange")
                st.caption("The model must include all DR grades and the not_eye class.")
            else:
                st.badge("Model available", color="green")
                st.caption(
                    "Model scores are not calibrated and are not diagnostic results."
                )

    st.subheader("Get started")
    workflow_columns = st.columns(3)
    workflow_cards = (
        (
            "1. Screen images",
            "Upload one or more retinal images and review the result table.",
        ),
        (
            "2. Train a model",
            "Provide labeled images for DR grades 0-4 and the not_eye class.",
        ),
        (
            "3. Review explanations",
            "Explore Grad-CAM, LIME, or SHAP to inspect model behavior.",
        ),
    )
    for column, (title, description) in zip(workflow_columns, workflow_cards):
        with column.container(border=True, height="stretch"):
            st.markdown(f"#### {title}")
            st.write(description)

    st.info(
        "This research app is not clinically validated. Do not use its outputs "
        "to make medical decisions."
    )

with tab_demo:
    st.subheader("Example demo")
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
    st.image(
        showcase_image,
        caption="Synthetic fundus sample",
        alt="Synthetic illustration of a retina",
        width=360,
    )
    if st.button("Run demonstration prediction"):
        st.success(f"Demo output: {CLASS_LABELS[str(showcase_class)]}")
        st.write(
            "The sample and output are synthetic. They do not represent a trained "
            "retinal classifier and must not be used for medical decisions."
        )

if checkpoint is None:
    with tab_upload:
        st.info("No trained model is available. Use the training tab to upload a dataset.")
else:
    image_size = int(checkpoint["image_size"])
    _, eval_transform = build_transforms(image_size)
    with tab_upload:
        if checkpoint.get("demo_training", False):
            st.error(
                "The installed checkpoint was trained on generated sample images, not "
                "retinal photographs. Its predictions are disabled. Train a new model "
                "with the training tab and a real labeled dataset before uploading photos."
            )
        elif set(class_names) != set(CLASS_LABELS):
            st.error(
                "This model is missing one or more DR grades or the not_eye class. "
                "Retrain with folders 0, 1, 2, 3, 4, and not_eye before using image screening."
            )
        else:
            uploaded_files = st.file_uploader(
                "Upload one or more retinal fundus images",
                type=("jpg", "jpeg", "png"),
                accept_multiple_files=True,
            )
            st.caption(
                "The model score is not a calibrated probability or a measure of "
                "diagnostic accuracy. Results are for research only."
            )
            if uploaded_files:
                images = []
                results = [None] * len(uploaded_files)
                for file_index, uploaded_file in enumerate(uploaded_files):
                    try:
                        with Image.open(uploaded_file) as opened_image:
                            images.append(
                                (
                                    file_index,
                                    uploaded_file.name,
                                    opened_image.convert("RGB"),
                                )
                            )
                    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                        results[file_index] = {
                            "Image": uploaded_file.name,
                            "Prediction": "Could not read image",
                            "Top model class": "",
                            "Model score": None,
                            "Status": "Invalid or unsupported image",
                        }

                predictions = predict_images(
                    model,
                    [(filename, image) for _, filename, image in images],
                    class_names,
                    eval_transform,
                    device,
                    MIN_MODEL_CONFIDENCE,
                )
                predictions_by_index = {}
                for (file_index, _, _), prediction in zip(images, predictions):
                    results[file_index] = prediction
                    predictions_by_index[file_index] = prediction
                results = [result for result in results if result is not None]
                st.subheader("Detection list")
                st.dataframe(
                    results,
                    hide_index=True,
                    alt="Model predictions and screening status for uploaded images",
                )
                csv_buffer = io.StringIO()
                columns = (
                    "Image",
                    "Prediction",
                    "Top model class",
                    "Model score",
                    "Status",
                )
                writer = csv.DictWriter(csv_buffer, fieldnames=columns)
                writer.writeheader()
                writer.writerows(results)
                st.download_button(
                    "Download detection list (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="retinopathy_detection_results.csv",
                    mime="text/csv",
                )

                if images:
                    selected_index = st.selectbox(
                        "Image to explain",
                        range(len(images)),
                        format_func=lambda index: images[index][1],
                    )
                    file_index, image_name, image = images[selected_index]
                    prediction = predictions_by_index[file_index]
                    predicted_label = prediction["Top model class"]
                    predicted_index = next(
                        index
                        for index, class_name in enumerate(class_names)
                        if CLASS_LABELS[class_name] == predicted_label
                    )
                    method = st.selectbox(
                        "Explanation method", ("Grad-CAM", "LIME", "SHAP")
                    )
                    if st.button("Generate explanation"):
                        try:
                            if method == "Grad-CAM":
                                heatmap = grad_cam(
                                    model, image, predicted_index, image_size, device
                                )
                                colored = Image.fromarray(
                                    (heatmap * 255).astype("uint8")
                                ).resize(image.size)
                                overlay = Image.blend(
                                    image, colored.convert("RGB"), alpha=0.4
                                )
                                st.image(
                                    overlay,
                                    caption=f"Grad-CAM attention overlay for {image_name}",
                                    alt="Grad-CAM attention overlay",
                                    width="stretch",
                                )
                            elif method == "LIME":
                                explanation = lime_explanation(
                                    model, image, image_size, device
                                )
                                st.image(
                                    explanation,
                                    caption=f"LIME superpixel explanation for {image_name}",
                                    alt="LIME superpixel explanation",
                                    width="stretch",
                                )
                            else:
                                attribution = shap_explanation(
                                    model, image, predicted_index, image_size, device
                                )
                                st.image(
                                    attribution,
                                    caption=f"SHAP attribution for {image_name}",
                                    alt="Mean absolute SHAP attribution",
                                    width="stretch",
                                )
                        except (ImportError, RuntimeError, ValueError) as error:
                            st.error(f"Could not generate the {method} explanation: {error}")

    if not checkpoint.get("demo_training", False):
        st.caption(
            f"Validation accuracy on the training split: {checkpoint['val_accuracy']:.1%}"
        )

with tab_training:
    st.subheader("Train on a retinal image dataset")
    st.info(
        "Training runs in the app process. The ZIP must contain real labeled images "
        "for all five DR grades plus a not_eye class. Streamlit Cloud restarts may "
        "remove models trained in the app."
    )
    dataset_archive = st.file_uploader(
        "Upload dataset ZIP",
        type=("zip",),
        help=(
            "The ZIP must contain train/ and val/ folders, each with class folders "
            "0, 1, 2, 3, 4, and not_eye."
        ),
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
