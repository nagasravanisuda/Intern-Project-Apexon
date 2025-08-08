# Databricks notebook source
name: Inference CI/CD Pipeline

on:
  push:
    branches: [ "main" ]
  workflow_dispatch:

jobs:
  inference-job:
    runs-on: ubuntu-latest

    env:
      AZURE_CONNECTION_STRING: ${{ secrets.AZURE_CONNECTION_STRING }}
      CONTAINER_NAME: demo
      BLOB_MODEL_PATH: model_artifacts/gradient_boosting_model.pkl
      BLOB_FEATURES_PATH: model_artifacts/feature_columns.pkl
      BLOB_SCALER_PATH: model_artifacts/scaler.pkl
      INPUT_CSV_PATH: inference_combined_real_synthetic.csv

    steps:
      - name: 📥 Checkout repo
        uses: actions/checkout@v3

      - name: 🐍 Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.9

      - name: 📦 Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas numpy scikit-learn matplotlib seaborn azure-storage-blob

      - name: 🚀 Run Inference Pipeline
        run: |
          python inference_pipeline.py \
            --connection_string "$AZURE_CONNECTION_STRING" \
            --container_name "$CONTAINER_NAME" \
            --model_blob "$BLOB_MODEL_PATH" \
            --features_blob "$BLOB_FEATURES_PATH" \
            --scaler_blob "$BLOB_SCALER_PATH" \
            --input_csv "$INPUT_CSV_PATH"

      - name: 📤 Upload output as artifact (optional)
        uses: actions/upload-artifact@v3
        with:
          name: inference-results
          path: inference_output_with_preds.csv
