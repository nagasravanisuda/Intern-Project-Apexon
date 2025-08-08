# Mortality Prediction using MIMIC-III Dataset with MLOps

This project focuses on predicting patient mortality using the MIMIC-III dataset by implementing a complete MLOps pipeline. It includes robust training and inference pipelines, use of synthetic data generation, feature engineering, and model deployment practices. The project uses CI/CD workflows to ensure streamlined and reproducible model lifecycle management.

---

## 🚀 Project Overview

The goal of this project is to build a machine learning model to predict mortality in ICU patients using the MIMIC-III dataset. The project is designed with MLOps best practices in mind and includes:

- 🔧 Training pipeline  
- 🔍 Inference pipeline  
- 🔁 Reusable code (Class-based implementation)  
- ✅ CI/CD workflows using GitHub Actions  

---

## 📂 Project Structure

```
├── training_pipeline/
│ └── train.py
├── inference_pipeline/
│ └── inference.py
├── cicd/
│ ├── training.yml
│ └── inference.yml
├── artifacts/
│ ├── model.pkl
│ ├── scaler.pkl
│ └── selected_features.pkl
├── requirements.txt
└── README.md

```

---

## 🏗️ Training Pipeline

The training pipeline includes:

- 📥 Data Ingestion from Azure Blob Storage (26 files, selected relevant ones)  
- 🧓 Age Column Creation from existing features  
- 🧹 Preprocessing (cleaning, encoding)  
- 🧬 Synthetic Data Generation using SDV's TVAE (deep learning-based tabular synthesis)  
- 🎯 Feature Selection using statistical and model-based techniques  
- 🤖 Model Training and artifact saving:  
  - Trained model (`model.pkl`)  
  - Scaler (`scaler.pkl`)  
  - Selected features (`selected_features.pkl`)  

These artifacts are saved for reuse during inference—no retraining is required unless explicitly needed.

---

## 🔎 Inference Pipeline

The inference pipeline includes:

- 🧩 Input data handling and preprocessing similar to training  
- 🔁 Reuse of trained model, scaler, and selected features  
- 📊 Prediction on unseen data  
- 🧱 Packaged as a reusable Python class for easy integration into other systems  

---

## ⚙️ CI/CD Workflows

CI/CD is implemented using GitHub Actions:

- `training.yml`: Automates training pipeline (optional retraining)  
- `inference.yml`: Validates and tests inference pipeline  
- `requirements.txt`: Includes all project dependencies for automated builds  

---

## 📦 Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```
## 📈 Results

The trained model is capable of generalizing well to unseen data, thanks to:

- Strong preprocessing pipeline  
- Synthetic data augmentation using TVAE  
- Proper feature selection and scaling  
- Artifact reuse in production  

---

## 🧠 Dataset

The model is trained using a subset of the MIMIC-III clinical dataset.  
*(Ensure you have proper credentials to access this dataset.)*

---

## 🧰 Technologies Used

- Python  
- scikit-learn, pandas, numpy  
- SDV (TVAE) for synthetic data generation  
- Azure Blob Storage  
- GitHub Actions for CI/CD  
- Pickle for artifact serialization  
