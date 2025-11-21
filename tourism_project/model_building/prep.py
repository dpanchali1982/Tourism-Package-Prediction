# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/dpanchali/Tourism-DataSet/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ==========================================
# Data Cleaning
# ==========================================
# Drop Unique Identifiers and Index columns that don't add predictive value
cols_to_drop = ['CustomerID', 'Unnamed: 0']
# Only drop if they actually exist in the dataframe
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=existing_cols_to_drop, inplace=True)

# Handle Missing Values (Simple imputation for robustness)
# Fills numeric NaNs with median and text NaNs with 'Unknown'
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = df[col].fillna(df[col].median())

# ==========================================
# Preprocessing (Encoding)
# ==========================================
# Identify categorical columns automatically (Gender, MaritalStatus, Occupation, etc.)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Encoding the following categorical columns: {categorical_cols}")

label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# ==========================================
# Splitting
# ==========================================
# 'ProdTaken' is the target variable (1 = Purchased, 0 = Not Purchased)
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split (80% Train, 20% Test)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save locally
print("Saving split files locally...")
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# ==========================================
# Upload to Hugging Face
# ==========================================
files_to_upload = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

print("Uploading files to Hugging Face...")
for file_path in files_to_upload:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path.split("/")[-1],  # just the filename
            repo_id='dpanchali/Tourism-DataSet',
            repo_type="dataset",
        )
        print(f"Uploaded: {file_path}")
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")

print("Data preparation pipeline completed.")
