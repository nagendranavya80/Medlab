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
- `src/` — data pipeline, model builder, and explainability helpers
- `artifacts/` — saved model and metadata after training

## Quick start

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train on Kaggle or Colab
- Upload the project to Kaggle or Colab.
- Mount your dataset and run:

```bash
python train.py --data_dir /path/to/fundus_dataset --epochs 10 --batch_size 32 --output_dir artifacts
```

For a quick smoke test with synthetic data:

```bash
python train.py --demo --epochs 2 --batch_size 8 --output_dir artifacts
```

To use this in a Colab notebook, open the included notebook template, upload the dataset into Google Drive, and then run the cells.

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
  val/
    0/
    1/
    2/
    3/
    4/
```

Classes are:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

## Explainability
The app produces:
- Grad-CAM heatmap to show model attention
- LIME saliency to highlight regions driving the prediction
- SHAP summary for model interpretation

## Deployment
To deploy online:
1. Push this repo to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Connect the repo and select `app.py` as the app entry point.
4. Add any required secrets, if needed.

## Clinical-use considerations
This project is intended as a decision-support tool and should be validated with expert clinicians and clinical performance metrics before use in patient care.
