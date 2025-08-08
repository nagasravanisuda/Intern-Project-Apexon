# Databricks notebook source
# Set the access key (if not already set)
spark.conf.set(
    "*****",
    "*****"
)

# Path to CSV in Azure Blob
csv_path = "***"

# Load CSV as Spark DataFrame
df = spark.read.option("header", True).csv(csv_path)
df.display()


# COMMAND ----------

# MAGIC %pip install sdv

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import os

# ---------------------------------------------
# Step 1: Load from Azure Blob (Spark to pandas)
# ---------------------------------------------
inference_path = "*****"
df = spark.read.option("header", True).csv(inference_path)
df = df.toPandas()
print("✅ Loaded inference data from Azure")

# ---------------------------------------------
# Step 2: Define preprocessing function
# ---------------------------------------------
def preprocess_inference_df(df):
    print("🔄 Preprocessing inference data...")

    # 1. Drop rows with >80% nulls
    threshold = int(0.8 * df.shape[1])
    df = df[df.isnull().sum(axis=1) < threshold]

    # 2. Drop unwanted columns
    cols_to_drop = [
        'subject_id', 'hadm_id', 'row_id', 'admittime', 'dischtime', 'deathtime',
        'edregtime', 'edouttime', 'dob', 'dod', 'dod_hosp', 'dod_ssn',
        'intime', 'outtime', 'ethnicity',
        'row_id_x', 'row_id_y', 'icustay_id'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 3. Handle nulls
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("datamiss")
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Handle outliers, skewness, and scaling
    numeric_cols = ['los', 'diagnosis_count', 'avg_chart_val', 'avg_lab_val']
    pt = PowerTransformer(method='yeo-johnson')
    scaler = MinMaxScaler()

    for col in numeric_cols:
        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        skew_val = skew(df[col])
        if abs(skew_val) > 0.5:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        else:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

        df[col] = np.clip(df[col], lower_bound, upper_bound)
        reshaped = df[col].values.reshape(-1, 1)
        df[col] = pt.fit_transform(reshaped).flatten()

    # Final scaling
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("✅ Preprocessing done.")
    return df

# ---------------------------------------------
# Step 3: Apply preprocessing and save
# ---------------------------------------------
df_clean = preprocess_inference_df(df)

# Create output directory if not exists
output_dir = "/dbfs/mortality"
os.makedirs(output_dir, exist_ok=True)

# Save preprocessed data
df_clean.to_csv(f"{output_dir}/preprocessed_inference.csv", index=False)
print("✅ Preprocessed data saved to /dbfs/mortality/preprocessed_inference.csv")


# COMMAND ----------

import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

# 2️⃣ Load your cleaned inference data
df = pd.read_csv("/dbfs/mortality/preprocessed_inference.csv")

# 3️⃣ Define metadata (auto-infer columns & types)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# 4️⃣ Initialize TVAE synthesizer
synthesizer = TVAESynthesizer(metadata)

# 5️⃣ Fit the synthesizer on your real data
synthesizer.fit(df)

# 6️⃣ Generate synthetic data (match real count or more)
num_rows_to_generate = 5000
synthetic_df = synthesizer.sample(num_rows_to_generate)

# 7️⃣ Combine real + synthetic
combined_df = pd.concat([df, synthetic_df], ignore_index=True)

# 8️⃣ Save final combined dataset
combined_df.to_csv("inference_combined_real_synthetic.csv", index=False)

print(f"✅ TVAE-based synthetic data generated and combined. Total rows = {combined_df.shape[0]}.")
print("📁 Saved to: inference_combined_real_synthetic.csv")


# COMMAND ----------

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

        # Evaluate if ground truth exists
        if "hospital_expire_flag" in df_output.columns:
            y_true = df_output["hospital_expire_flag"]
            print("\n📊 Classification Report:\n")
            print(classification_report(y_true, y_pred, zero_division=0))

            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.show()

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)

            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ 'hospital_expire_flag' not found — skipping evaluation.")

        df_output.to_csv(output_path, index=False)
        print(f"✅ Inference results saved to: {output_path}")
        return df_output


# ========== ✅ Run only if executed directly (not when imported) ==========
if __name__ == "__main__":
    connection_string = "*****"
    container_name = "****"
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
