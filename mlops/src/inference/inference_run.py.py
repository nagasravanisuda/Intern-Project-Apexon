# Databricks notebook source
# MAGIC %pip install azure-storage-blob
# MAGIC

# COMMAND ----------

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score
)
from azure.storage.blob import BlobServiceClient

class InferencePipeline:
    def __init__(self, connection_string, container_name, blob_model_path, blob_features_path, blob_scaler_path):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_model_path = blob_model_path
        self.blob_features_path = blob_features_path
        self.blob_scaler_path = blob_scaler_path
        self.model = None
        self.feature_list = None
        self.scaler = None

    def download_blob_to_file(self, blob_name, local_path):
        blob_service = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service.get_container_client(self.container_name)
        blob_client = container_client.get_blob_client(blob_name)
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        print(f"✅ Downloaded {blob_name} to {local_path}")

    def load_model_and_features(self):
        self.download_blob_to_file(self.blob_model_path, "model.pkl")
        self.download_blob_to_file(self.blob_features_path, "features.pkl")
        self.download_blob_to_file(self.blob_scaler_path, "scaler.pkl")

        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("features.pkl", "rb") as f:
            self.feature_list = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        print("✅ Model, features, and scaler loaded.")

    def run_inference(self, input_csv_path, output_path="inference_output_with_preds.csv"):
        print("📥 Loading inference data...")
        df = pd.read_csv(input_csv_path)

        # Drop unnecessary columns
        drop_cols = ['subject_id', 'hadm_id', 'row_id', 'admittime', 'dischtime', 'deathtime',
                     'edregtime', 'edouttime', 'dob', 'dod', 'dod_hosp', 'dod_ssn',
                     'intime', 'outtime', 'ethnicity', 'row_id_x', 'row_id_y', 'icustay_id']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

        # Fill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("datamiss")
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)

        # Optional feature engineering
        religion_cols = [col for col in df.columns if col.startswith("religion_")]
        if religion_cols:
            df["religion_count"] = df[religion_cols].sum(axis=1)

        # One-hot encoding
        df = pd.get_dummies(df, drop_first=True)

        # Add missing features from training
        for col in self.feature_list:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_list]  # Ensure column order

        # Scale
        X_scaled = self.scaler.transform(df)

        # Predict
        print("🧠 Running predictions...")
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]

        # Save output
        df_output = pd.read_csv(input_csv_path).loc[df.index].copy()
        df_output["predicted_label"] = y_pred
        df_output["predicted_probability"] = y_prob

        df_output.to_csv(output_path, index=False)
        print(f"✅ Inference results saved to: {output_path}")
        return df_output


# ========== ✅ Run only if executed directly (not when imported) ==========
if __name__ == "__main__":
    connection_string = "DefaultEndpointsProtocol=https;AccountName=nagasravani;AccountKey=RUaWe24nwtC9HnWC+iooZmTtYf3KWEGyk08L2NaUZpPYJda7BABtP0MqPHXbsFwJ+HtWjwTKkCic+AStx/gUUA==;EndpointSuffix=core.windows.net"
    container_name = "demo"
    blob_model_path = "model_artifacts/gradient_boosting_model.pkl"
    blob_features_path = "model_artifacts/feature_columns.pkl"
    blob_scaler_path = "model_artifacts/scaler.pkl"
    input_csv_path = "inference_combined_real_synthetic.csv"

    pipeline = InferencePipeline(
        connection_string,
        container_name,
        blob_model_path,
        blob_features_path,
        blob_scaler_path
    )

    pipeline.load_model_and_features()
    result_df = pipeline.run_inference(input_csv_path)
