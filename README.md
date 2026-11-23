# Diabetic Retinopathy Diagnostic Tool

This project builds a trustworthy and explainable diabetic retinopathy diagnostic pipeline using a pretrained ResNet50 backbone and a Streamlit web app.

## Features
- Fundus image preprocessing and labeling
- Transfer learning with pretrained ResNet50
- Training and validation pipeline with class imbalance awareness
- Explainability using Grad-CAM, LIME, and SHAP
- Local deployment with Streamlit
- Cloud-ready project structure for Kaggle or Google Colab

## Folder structure
- `train.py` — training entry point
- `app.py` — local web app
- `src/data.py` — image-folder loading, transforms, and class balancing
- `src/model.py` — ResNet50 model builder
- `src/explainability.py` — Grad-CAM, LIME, and SHAP helpers
- `artifacts/` — best model checkpoint and validation metadata after training
- `requirements.txt` — Python dependencies

## Quick start

### 1) Create a virtual environment
```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Train on Kaggle or Colab
- Upload the project to Kaggle or Colab.
- Mount your dataset and run:

```bash
python train.py --data_dir /path/to/fundus_dataset --epochs 10 --batch_size 32 --output_dir artifacts
```

On Windows, use a path such as `C:\data\fundus_dataset`. The first normal training
run downloads the pretrained ResNet50 ImageNet weights. Add `--no_pretrained`
to initialize the model without downloading them.

For a small functional demo using generated images (not real medical data):

```bash
python train.py --demo --epochs 1 --batch_size 4 --image_size 64 --output_dir artifacts
```

The demo trains a randomly initialized model and is only intended to check that
the training pipeline runs; its metrics have no clinical meaning.

### 3) Run the Streamlit app locally
```bash
streamlit run app.py
```

## Dataset guidance
Use a labeled fundus image dataset such as:
- EyePACS / Kaggle APTOS 2019
- Messidor
- IDRID

Expected folder organization:
```text
data/
  train/
    0/
    1/
    2/
    3/
    4/
    not_eye/  # Optional; recommended for rejecting unrelated uploads
  val/
    0/
    1/
    2/
    3/
    4/
    not_eye/  # Include when using train/not_eye
```

Classes are:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR
- `not_eye` (optional): images that are not retinal fundus photos

The `not_eye` class is optional for backwards compatibility, but strongly
recommended. Include diverse non-fundus examples (people, objects, documents,
other medical images, and poor-quality uploads) in both splits to teach the model
when to reject an image. The app also rejects low-confidence outputs, but neither
that check nor the learned `not_eye` class guarantees detection of every unrelated
image. Retrain after adding this class.

## Explainability
The app can generate Grad-CAM overlays, LIME superpixel explanations, and SHAP
attribution maps. Explanations are computationally expensive and describe model
behavior, not clinical reasoning.

## Deployment
To deploy on Streamlit Community Cloud:
1. Push this repo to GitHub, including the model checkpoint in `artifacts/`.
2. Sign in to Streamlit Community Cloud.
3. Connect the repo and select `app.py` as the app entry point.
4. Choose Python 3.11 and deploy.

## Clinical-use considerations
This project is intended as a decision-support tool and should be validated with expert clinicians and clinical performance metrics before use in patient care.
