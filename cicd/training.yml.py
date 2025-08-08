# Databricks notebook source
name: Training CI/CD Pipeline

on:
  push:
    branches: [ "main" ]
  workflow_dispatch:

jobs:
  training-job:
    runs-on: ubuntu-latest

    env:
      AZURE_CONNECTION_STRING: ${{ secrets.AZURE_CONNECTION_STRING }}
      CONTAINER_NAME: demo
      LOCAL_MODEL_DIR: model_output
      BLOB_PREFIX: model_artifacts/

    steps:
      - name: 📥 Checkout repository
        uses: actions/checkout@v3

      - name: 🐍 Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.10

      - name: 📦 Install Python dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: 🚀 Run Training Script
        run: |
          python mlops/src/training/train_model.py

      - name: ☁️ Upload Artifacts to Azure Blob Storage
        run: |
          python mlops/scripts/upload_to_azure_blob.py
