from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.data import CLASS_LABELS, build_transforms
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


st.set_page_config(page_title="Diabetic Retinopathy Tool", page_icon="👁️")
st.title("Diabetic Retinopathy Diagnostic Tool")
st.warning(
    "Research and decision-support demo only. It is not a diagnosis and must not be "
    "used to guide patient care."
)

if not CHECKPOINT.is_file():
    st.info("No trained model found. Train one first with `python train.py --data_dir data`.")
    st.stop()

model, checkpoint, device = load_model(str(CHECKPOINT), CHECKPOINT.stat().st_mtime)
if checkpoint.get("demo_training", False):
    st.error(
        "DEMO MODEL: predictions are not clinically meaningful. This model was trained "
        "only on generated images, not real retinal photographs."
    )

image_size = int(checkpoint["image_size"])
_, eval_transform = build_transforms(image_size)
uploaded_file = st.file_uploader("Upload a retinal fundus image", type=("jpg", "jpeg", "png"))

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
            "fundus image, or it may be too different from the training images. "
            "Please upload a clear retinal fundus photo."
        )
        st.stop()
    if predicted_class == "not_eye":
        st.error(
            "This does not appear to be a retinal fundus image. Please upload a "
            "clear retinal fundus photo."
        )
        st.stop()
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
                colored = colored.convert("RGB")
                overlay = Image.blend(image, colored, alpha=0.4)
                st.image(overlay, caption="Grad-CAM attention overlay", use_container_width=True)
            elif method == "LIME":
                explanation = lime_explanation(model, image, image_size, device)
                st.image(explanation, caption="LIME superpixel explanation", use_container_width=True)
            else:
                attribution = shap_explanation(model, image, predicted_index, image_size, device)
                st.image(
                    attribution,
                    caption="Mean absolute SHAP attribution",
                    clamp=True,
                    use_container_width=True,
                )
        except (ImportError, RuntimeError, ValueError) as error:
            st.error(f"Could not generate the {method} explanation: {error}")

st.caption(
    f"Validation accuracy on the training split: {checkpoint['val_accuracy']:.1%}"
)
