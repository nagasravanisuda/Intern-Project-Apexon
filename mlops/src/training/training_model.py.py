# Databricks notebook source
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install kagglehub[pandas-datasets]
# MAGIC %pip install kaggle     
# MAGIC

# COMMAND ----------

import os
from azure.storage.blob import BlobServiceClient

dataset_slug = "asjad99/mimiciii"
download_dir = "/tmp/mimic_download"
kaggle_dbfs_path = "**********"

connection_string = "********"
container_name = "******"

print(" Setting up Kaggle credentials...")

os.makedirs("/root/.kaggle", exist_ok=True)

if os.path.exists("/root/.kaggle/kaggle.json"):
    os.remove("/root/.kaggle/kaggle.json")

dbutils.fs.cp(
    kaggle_dbfs_path,
    "file:/root/.kaggle/kaggle.json"
)

os.system("chmod 600 /root/.kaggle/kaggle.json")
os.environ["KAGGLE_CONFIG_DIR"] = "/root/.kaggle"


print("\n Downloading Kaggle dataset...")
os.makedirs(download_dir, exist_ok=True)

download_result = os.system(f"kaggle datasets download -d {dataset_slug} -p {download_dir} --unzip")

if os.listdir(download_dir):
    print(f" Dataset downloaded and unzipped to: {download_dir}")
else:
    raise Exception(" Download failed. Check Kaggle credentials or dataset slug.")

print("\n Uploading files to Azure Blob Storage...")
blob_service = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service.get_container_client(container_name)

if not container_client.exists():
    container_client.create_container()
    print(f" Created container: {container_name}")

for file_name in os.listdir(download_dir):
    file_path = os.path.join(download_dir, file_name)
    if os.path.isfile(file_path):
        print(f" Uploading: {file_name}")
        with open(file_path, "rb") as data:
            blob_client = container_client.get_blob_client(file_name)
            blob_client.upload_blob(data, overwrite=True)
    else:
        print(f" Skipping directory: {file_name}")

print("\n All files uploaded to Azure Blob Storage!")


# COMMAND ----------

spark.conf.set(
  "******",
  "******"
)


# COMMAND ----------

dbutils.fs.ls("*******")


# COMMAND ----------

csv_files = dbutils.fs.ls("*******")
display(csv_files)



# COMMAND ----------

from pyspark.sql.functions import col, avg, countDistinct, datediff, to_date, lit, when, first
from pyspark.sql.types import IntegerType

path = "*******"

admissions = spark.read.csv(f"{path}/ADMISSIONS.csv", header=True, inferSchema=True)
patients = spark.read.csv(f"{path}/PATIENTS.csv", header=True, inferSchema=True)
icustays = spark.read.csv(f"{path}/ICUSTAYS.csv", header=True, inferSchema=True)
chartevents = spark.read.csv(f"{path}/CHARTEVENTS.csv", header=True, inferSchema=True)
labevents = spark.read.csv(f"{path}/LABEVENTS.csv", header=True, inferSchema=True)
diagnoses_icd = spark.read.csv(f"{path}/DIAGNOSES_ICD.csv", header=True, inferSchema=True)

chartevents_small = chartevents.filter(col("valuenum").isNotNull()) \
    .select("subject_id", "hadm_id", "valuenum", "valueuom")

labevents_small = labevents.filter(col("valuenum").isNotNull()) \
    .select("subject_id", "hadm_id", "valuenum")

diagnoses_small = diagnoses_icd.select("subject_id", "hadm_id", "icd9_code")

chart_agg = chartevents_small.groupBy("subject_id", "hadm_id").agg(
    avg("valuenum").alias("avg_chart_val"),
    first("valueuom", ignorenulls=True).alias("valueuom")  
)

lab_agg = labevents_small.groupBy("subject_id", "hadm_id").agg(
    avg("valuenum").alias("avg_lab_val")
)

diag_count = diagnoses_small.groupBy("subject_id", "hadm_id").agg(
    countDistinct("icd9_code").alias("diagnosis_count")
)

patients = patients.withColumn("dob", to_date("dob"))
admissions = admissions.withColumn("admittime", to_date("admittime"))

admissions = admissions.join(
    patients.select("subject_id", "dob", "gender"),
    on="subject_id",
    how="left"
)

admissions = admissions.withColumn(
    "age",
    (datediff(col("admittime"), col("dob")) / lit(365.25)).cast(IntegerType())
)
admissions = admissions.withColumn("age", when(col("age") > 120, 120).otherwise(col("age")))

combined = admissions \
    .join(icustays, ["subject_id", "hadm_id"], "left") \
    .join(chart_agg, ["subject_id", "hadm_id"], "left") \
    .join(lab_agg, ["subject_id", "hadm_id"], "left") \
    .join(diag_count, ["subject_id", "hadm_id"], "left")

combined_pd = combined.toPandas()
combined_pd.to_csv("combined.csv", index=False)

print("✅ Saved combined.csv with age, gender, avg_chart_val, and valueuom.")


# COMMAND ----------

from azure.storage.blob import BlobServiceClient
import os

# Azure details
connection_string = "******"
container_name = "****"
local_file_path = "*****"
blob_folder = "*****"
blob_name = "******"  # like a path

# ✅ Create blob service client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# ✅ Get container client
container_client = blob_service_client.get_container_client(container_name)
if not container_client.exists():
    container_client.create_container()
    print(f"🆕 Created container: {container_name}")

# ✅ Upload file to folder in Blob (folder is part of blob name)
with open(local_file_path, "rb") as data:
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True)
    print(f"✅ Uploaded '{local_file_path}' to 'demo/{blob_folder}/' as '{blob_name}'")


# COMMAND ----------

import pandas as pd   
combined_pd = pd.read_csv("combined.csv")
print(" Sample Rows:")
display(combined_pd.head())
print("\n DataFrame Shape:")
display(combined_pd.shape)
print("\n Data Description:")
print(combined_pd.describe(include='all'))


# COMMAND ----------

import pandas as pd

df = pd.read_csv("combined.csv")

discharge_location_counts = df['discharge_location'].value_counts().reset_index()
discharge_location_counts.columns = ['discharge_location', 'count']
print(" Discharge Location Counts:")
display(discharge_location_counts)

gender_counts = df['gender'].value_counts().reset_index()
gender_counts.columns = ['gender', 'count']
print("\n Gender Counts:")
display(gender_counts)

hospital_expire_flag_counts = df['hospital_expire_flag'].value_counts().reset_index()
hospital_expire_flag_counts.columns = ['hospital_expire_flag', 'count']
print("\n Hospital Expiry Flag Counts:")
display(hospital_expire_flag_counts)

age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
age_labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']
df['age_binned'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
age_binned_counts = df['age_binned'].value_counts().sort_index().reset_index()
age_binned_counts.columns = ['Age Range', 'count']
print("\n Binned Age Distribution:")
display(age_binned_counts)

los_bins = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 100]
los_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-15', '15-20', '20-25', '25-30', '30+']
df['los_binned'] = pd.cut(df['los'], bins=los_bins, labels=los_labels, right=False)
los_binned_counts = df['los_binned'].value_counts().sort_index().reset_index()
los_binned_counts.columns = ['LOS Range (days)', 'count']
print("\n Binned Length of Stay (LOS):")
display(los_binned_counts)



# COMMAND ----------

import pandas as pd

# 1. Load the combined dataset
df = pd.read_csv("combined.csv")

# 2. Drop rows with >80% null values
threshold = int(0.8 * df.shape[1])
df = df[df.isnull().sum(axis=1) < threshold]

# 3. Drop unwanted ID and timestamp columns
cols_to_drop = [
    'subject_id', 'hadm_id', 'row_id', 'admittime', 'dischtime', 'deathtime',
    'edregtime', 'edouttime', 'dob', 'dod', 'dod_hosp', 'dod_ssn',
    'intime', 'outtime', 'ethnicity',
    'row_id_x', 'row_id_y', 'icustay_id'
]
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# 4. Replace nulls:
#    - string columns → "datamiss"
#    - numeric columns → 0
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("datamiss")
    else:
        df[col] = df[col].fillna(0)

# 5. Save cleaned data
df.to_csv("final_cleaned_combined.csv", index=False)

print("✅ Cleaning complete. File saved as 'final_cleaned_combined.csv'.")


# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

# --- Load Data ---
df = pd.read_csv("final_cleaned_combined.csv")  
numeric_cols =['los','diagnosis_count','avg_chart_val','avg_lab_val']

for col in numeric_cols:
    print(f"\n🔍 Processing column: {col}")

    # 1️⃣ Skewness BEFORE
    skew_val = skew(df[col])
    print(f"🔸 Skewness BEFORE: {skew_val:.2f}")

    # 2️⃣ Boxplot BEFORE outlier handling
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"{col} - BEFORE Outlier Replacement")
    plt.show()

    # 3️⃣ Outlier Detection & Replacement
    if abs(skew_val) > 0.5:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        print("📏 Using IQR Method")
    else:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        print("📏 Using Z-Score Method")
    
    # Replace outliers with clipped values
    df[col] = np.clip(df[col], lower_bound, upper_bound)

    # 4️⃣ Boxplot AFTER outlier replacement
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"{col} - AFTER Outlier Replacement")
    plt.show()

    # 5️⃣ Histogram BEFORE Yeo-Johnson
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} - BEFORE Yeo-Johnson Transformation")
    plt.show()

    # 6️⃣ Apply Yeo-Johnson Transformation
    reshaped = df[col].values.reshape(-1, 1)
    pt = PowerTransformer(method='yeo-johnson')
    transformed = pt.fit_transform(reshaped)
    df[col] = transformed.flatten()

    # 7️⃣ Histogram AFTER Yeo-Johnson
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} - AFTER Yeo-Johnson Transformation")
    plt.show()

    # 8️⃣ Skewness AFTER
    new_skew = skew(df[col])
    print(f"✅ Skewness AFTER Yeo-Johnson: {new_skew:.2f}")

# --- Normalize Numeric Columns ---
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# --- Save Final Output ---
df.to_csv("final_selected_df_cleaned.csv", index=False)
print("\n✅ Final cleaned data saved as: final_selected_df_cleaned.csv")   




# COMMAND ----------

# MAGIC %pip install sdv

# COMMAND ----------

pip install imblearn

# COMMAND ----------

# 1️⃣ Import libraries
import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

# 2️⃣ Load your original cleaned data
df = pd.read_csv("final_selected_df_cleaned.csv")

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
combined_df.to_csv("final_combined_real_synthetic.csv", index=False)

print(f"✅ TVAE-based synthetic data generated and combined. Total rows = {combined_df.shape[0]}.")
print("📁 Saved to: final_combined_real_synthetic.csv")


# COMMAND ----------

# 🔁 Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 📥 Load Data
df = pd.read_csv("final_combined_real_synthetic.csv")
target_col = 'hospital_expire_flag'

# 🧹 Drop ID/Time Columns including dob
drop_cols = [col for col in df.columns if
             col.lower() in ['subject_id', 'hadm_id', 'row_id','first_wardid','last_wardid','row_id.1'] or
             col.lower().endswith('_id') or
             'time' in col.lower() or
             col.lower() in ['admittime', 'dischtime', 'deathtime', 'dob']]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# 🧼 Handle Missing Values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("datamiss")
    else:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

# 🔄 One-hot Encoding
df = pd.get_dummies(df, drop_first=True)

# 🎯 Features and Target
y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
X = df.drop(columns=[target_col])

# 📊 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🌲 Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_scores = pd.Series(rf.feature_importances_, index=X.columns, name="RF")

# 🔍 ANOVA F-test
anova = SelectKBest(score_func=f_classif, k='all').fit(X_train, y_train)
anova_scores = pd.Series(anova.scores_, index=X.columns, name="ANOVA")

# ℹ️ Mutual Information
mi = SelectKBest(score_func=mutual_info_classif, k='all').fit(X_train, y_train)
mi_scores = pd.Series(mi.scores_, index=X.columns, name="MI")

# 📌 Top 25 from Each
top_rf = rf_scores.sort_values(ascending=False).head(25).index.tolist()
top_anova = anova_scores.sort_values(ascending=False).head(25).index.tolist()
top_mi = mi_scores.sort_values(ascending=False).head(25).index.tolist()

# ✅ Final Union
final_features = sorted(set(top_rf + top_anova + top_mi))
print(f"✅ Total unique selected features: {len(final_features)}")

# 🧾 Selected Feature DataFrame
df_selected = df[final_features + [target_col]]

# 🔁 Correlation with Target
correlations = df_selected.corr()[target_col].drop(target_col).sort_values(ascending=False)
print("\n📈 Top features correlated with target:\n", correlations.head(10))

# 🌡️ Heatmap of Top 20 Correlated Features
top20_corr_features = correlations.head(20).index.tolist()
plt.figure(figsize=(20, 14))
sns.heatmap(
    df_selected[top20_corr_features + [target_col]].corr(),
    annot=True, fmt=".2f", cmap='coolwarm', center=0,
    annot_kws={"size": 10}
)
plt.title("🔍 Correlation Heatmap - Top 20 Selected Features")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 📊 Bar Plots

# plt.figure(figsize=(8, 6))
rf_scores[top_rf].sort_values().plot(kind='barh', title='🔍 Random Forest Feature Importance')
for index, value in enumerate(rf_scores[top_rf].sort_values()):
    plt.text(value + 0.0005, index, f"{value:.3f}", va='center', fontsize=8)
plt.tight_layout()
plt.show()

# ANOVA F-test
plt.figure(figsize=(8, 6))
anova_scores[top_anova].sort_values().plot(kind='barh', title='📊 ANOVA F-test Scores')
for index, value in enumerate(anova_scores[top_anova].sort_values()):
    plt.text(value + 0.5, index, f"{value:.1f}", va='center', fontsize=8)
plt.tight_layout()
plt.show()

# Mutual Information
plt.figure(figsize=(8, 6))
mi_scores[top_mi].sort_values().plot(kind='barh', title='ℹ️ Mutual Information Scores')
for index, value in enumerate(mi_scores[top_mi].sort_values()):
    plt.text(value + 0.001, index, f"{value:.3f}", va='center', fontsize=8)
plt.tight_layout()
plt.show()


# 📸 Pairplot of Top 6 Features + Target
top6_corr_features = correlations.head(6).index.tolist()
pairplot_df = df_selected[top6_corr_features + [target_col]]
sns.pairplot(
    pairplot_df,
    hue=target_col,
    palette="Set1",
    diag_kind="kde",
    plot_kws={'alpha': 0.6}
)
plt.suptitle("📊 Pairplot of Top 6 Correlated Features", y=1.02)
plt.tight_layout()
plt.show()

# 💾 Save to DBFS
output_path = "/dbfs/mortality"
os.makedirs(output_path, exist_ok=True)  # ✅ Create the directory if it doesn't exist
df_selected.to_csv(f"{output_path}/union_selected_features.csv", index=False)
print("✅ Saved: /dbfs/mortality/union_selected_features.csv")


# COMMAND ----------

!pip install lightgbm


# COMMAND ----------

# MAGIC
# MAGIC %pip install xgboost catboost
# MAGIC

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("/dbfs/mortality/union_selected_features.csv")
target_col = 'hospital_expire_flag'
X = df.drop(columns=[target_col])
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE (train):")
print(pd.Series(y_train).value_counts(normalize=True))
print("\nAfter SMOTE (train):")
print(pd.Series(y_train_res).value_counts(normalize=True))

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "LightGBM": lgb.LGBMClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

results = {}

def plot_roc_curve(model, X_test, y_test, model_name):
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
    return auc_score

plt.figure(figsize=(10, 6))
plt.title("ROC Curve - Base Models")

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1}

    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    try:
        plot_roc_curve(model, X_test, y_test, name)
    except Exception as e:
        print(f"❌ Could not plot ROC for {name}: {e}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

print("\n🔧 Starting Grid Search for Tuned Models")
tuned_results = {}

rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train_res, y_train_res)
best_rf = rf_grid.best_estimator_
print("\nBest Random Forest:", rf_grid.best_params_)

gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 10]}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
gb_grid.fit(X_train_res, y_train_res)
best_gb = gb_grid.best_estimator_
print("\nBest Gradient Boosting:", gb_grid.best_params_)

lgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [5, 10, 15], 'num_leaves': [31, 50, 70], 'min_child_samples': [20, 30]}
lgb_grid = GridSearchCV(lgb.LGBMClassifier(random_state=42), lgb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
lgb_grid.fit(X_train_res, y_train_res)
best_lgb = lgb_grid.best_estimator_
print("\nBest LightGBM:", lgb_grid.best_params_)

plt.figure(figsize=(10, 6))
plt.title("RoC Curve - Tuned Models")
tuned_models = {"Random Forest": best_rf, "Gradient Boosting": best_gb, "LightGBM": best_lgb}

for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tuned_results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1}

    print(f"\n Model: {name} (Tuned)")
    print(f" Accuracy: {acc:.4f}, Precision: {prec:.4f},  Recall: {rec:.4f},  F1-score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    try:
        plot_roc_curve(model, X_test, y_test, name + " (Tuned)")
    except Exception as e:
        print(f" Could not plot ROC for {name}: {e}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC 1
# MAGIC

# COMMAND ----------

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE


class MortalityPredictionPipeline:
    def __init__(self, data_path, target_col, save_dir="/dbfs/mortality/model_output"):
        self.data_path = data_path
        self.target_col = target_col
        self.save_dir = save_dir
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.results = {}

        os.makedirs(self.save_dir, exist_ok=True)

    def load_data(self):
        print("🔄 Loading data...")
        self.df = pd.read_csv(self.data_path)

        # Optional: Add religion_count if religion_* columns exist
        religion_cols = [col for col in self.df.columns if col.startswith("religion_")]
        if religion_cols:
            self.df['religion_count'] = self.df[religion_cols].sum(axis=1)

        # Clean data
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]
        print("✅ Data loaded and preprocessed")

    def preprocess_data(self):
        print("🔄 Scaling and splitting data...")
        self.X_scaled = self.scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        self.X_train_res, self.y_train_res = self.smote.fit_resample(self.X_train, self.y_train)
        print("✅ Data balanced with SMOTE")

    def train_model(self):
        print("🎯 Training Gradient Boosting model...")
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.model.fit(self.X_train_res, self.y_train_res)
        print("✅ Model training complete")

    def evaluate_model(self):
        print("📈 Evaluating model...")
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        self.results = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "ROC AUC": roc_auc
        }

        print("\n📊 Classification Report")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        print(f"✅ Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-score: {f1:.4f} | AUC: {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "confusion_matrix.png"))
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def save_model_and_predictions(self):
        print("💾 Saving model, scaler, metrics, and feature list to DBFS...")

        with open(os.path.join(self.save_dir, "gradient_boosting_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        with open(os.path.join(self.save_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(self.save_dir, "model_metrics.pkl"), "wb") as f:
            pickle.dump(self.results, f)

        with open(os.path.join(self.save_dir, "feature_columns.pkl"), "wb") as f:
            pickle.dump(self.X.columns.tolist(), f)

        print(f"✅ Model, scaler, metrics, and features saved in: {self.save_dir}")


# ✅ USAGE: Train the pipeline
if __name__ == "__main__":
    pipeline = MortalityPredictionPipeline(
        data_path="/dbfs/mortality/union_selected_features.csv",  # 👈 Update your path here
        target_col="hospital_expire_flag",
        save_dir="/dbfs/mortality/model_output"  # 👈 Output folder for model + plots
    )
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.train_model()
    pipeline.evaluate_model()
    pipeline.save_model_and_predictions()


# COMMAND ----------

from azure.storage.blob import BlobServiceClient
import os

# Set up Azure Blob Storage connection
connection_string = "******"
container_name = "*****"
local_model_dir = "******"
blob_prefix = "*****"  # folder inside your container

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Create container if not exists
if not container_client.exists():
    container_client.create_container()

# Upload .pkl and other files
for file_name in os.listdir(local_model_dir):
    if file_name.endswith(".pkl") or file_name.endswith(".png"):
        local_file_path = os.path.join(local_model_dir, file_name)
        blob_path = blob_prefix + file_name
        print(f"Uploading {file_name} to Azure Blob: {blob_path}")

        with open(local_file_path, "rb") as data:
            blob_client = container_client.get_blob_client(blob_path)
            blob_client.upload_blob(data, overwrite=True)

print("✅ All model artifacts uploaded to Azure Blob Storage.")
