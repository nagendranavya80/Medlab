# Diabetic Retinopathy Diagnostic Tool

This project builds a trustworthy and explainable diabetic retinopathy diagnostic pipeline using a pretrained ResNet50 backbone and a Streamlit web app.

## Features
- Professional home page with model readiness and workflow navigation
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
    not_eye/
  val/
    0/
    1/
    2/
    3/
    4/
    not_eye/
```

Classes are:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR
- `not_eye`: images that are not retinal fundus photos

The app requires all six classes, including `not_eye`, before it enables image
screening. Include diverse non-fundus examples (people, objects, documents, other
medical images, and poor-quality uploads) in both splits to teach the model when
to reject an image. The app rejects low-score outputs, but neither that check nor
the learned `not_eye` class guarantees detection of every unrelated image.

The included checkpoint is a synthetic demo model trained on generated images,
not retinal photographs. The app disables predictions with this checkpoint.
Train with a real, labeled dataset before using image prediction. The image
prediction tab accepts multiple JPG/PNG images and provides a downloadable CSV
detection list.

### Preparing APTOS images
The [APTOS 2019 Kaggle competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
provides labeled DR images. Sign in to Kaggle and accept the competition rules to
download the data. Use its labeled `train_images/` and `train.csv`; do not use
unlabeled test images for training.

Collect a separate folder of non-retinal images that you have permission to use.
Include varied everyday photos, documents, and other images; do not use patient
photos without appropriate permission. Then prepare the six-class train/validation
folders locally:

```powershell
python prepare_aptos_dataset.py `
  --images-dir "C:\data\aptos\train_images" `
  --labels-csv "C:\data\aptos\train.csv" `
  --not-eye-dir "C:\data\not_eye" `
  --output-dir "C:\data\prepared"
Compress-Archive -Path "C:\data\prepared\train","C:\data\prepared\val" `
  -DestinationPath "C:\data\retinopathy_training.zip"
```

Upload `retinopathy_training.zip` in the app's **Train a real model** tab. The
preparation script splits each DR grade and `not_eye` class between train and
validation. Training on public research data is still not clinical validation.

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
