# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import subprocess
import time
from pyngrok import ngrok, conf

# ==========================================
# 1. Configuration & Setup
# ==========================================

# Configure Ngrok Auth Token
# Using conf.get_default() is safer for newer pyngrok versions
NGROK_TOKEN = "35GpiJlxPKUMCe5ZHuuzO2HZOnK_4nRL2j8BabB8wgB1bP3s5" 
conf.get_default().auth_token = NGROK_TOKEN

# Start MLflow UI on port 5000 in the background
# We use subprocess to run it as a separate non-blocking process
subprocess.Popen(["mlflow", "ui", "--port", "5000"])

# Wait a moment for MLflow to start
time.sleep(5)

# Create public tunnel
try:
    # Connect to port 5000
    public_url = ngrok.connect(5000).public_url
    print("MLflow UI is available at:", public_url)
    # Set the tracking URL for MLflow
    mlflow.set_tracking_uri(public_url)
except Exception as e:
    print(f"Error connecting to Ngrok: {e}")
    print("Falling back to local tracking URI")
    mlflow.set_tracking_uri("http://localhost:5000")

# Set the name for the experiment
mlflow.set_experiment("tourism-prediction-experiment")

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

print("Configuration loaded. Starting model training pipeline...")

# ==========================================
# 2. Load Processed Data from Hugging Face
# ==========================================

Xtrain_path = "hf://datasets/dpanchali/Tourism-DataSet/Xtrain.csv"
Xtest_path = "hf://datasets/dpanchali/Tourism-DataSet/Xtest.csv"
ytrain_path = "hf://datasets/dpanchali/Tourism-DataSet/ytrain.csv"
ytest_path = "hf://datasets/dpanchali/Tourism-DataSet/ytest.csv"

# Load into Pandas DataFrames
print("Loading data...")
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Ensure target is a 1D array
ytrain = ytrain.values.ravel()
ytest = ytest.values.ravel()

print("Data loaded successfully.")

# ==========================================
# 3. Preprocessing Pipeline Setup
# ==========================================
print("Setting up preprocessing pipeline...")

# Define column groups based on tourism.csv structure
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
    'MaritalStatus', 'Designation', 'CityTier', 'Passport', 'OwnCar'
]

# Ensure these columns exist in Xtrain (intersect to be safe)
categorical_features = [c for c in categorical_features if c in Xtrain.columns]
numeric_features = [c for c in Xtrain.columns if c not in categorical_features]

print(f"Categorical Features: {categorical_features}")

# Create Column Transformer
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    (StandardScaler(), numeric_features),
    remainder='passthrough'
)

# ==========================================
# 4. Class Imbalance Handling
# ==========================================
# Calculate scale_pos_weight for XGBoost
num_neg = sum(ytrain == 0)
num_pos = sum(ytrain == 1)
scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# ==========================================
# 5. Model Definition & Hyperparameter Tuning
# ==========================================
print("Initializing XGBoost and GridSearchCV...")

# Define the XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight, # Handle imbalance
    random_state=42
)

# Create the full pipeline: Preprocessor -> Model
model_pipeline = make_pipeline(preprocessor, xgb_clf)

# Define Hyperparameter Grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.1],
    'xgbclassifier__subsample': [0.8]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Start MLflow Run
with mlflow.start_run():
    print("Training model pipeline with Grid Search...")
    grid_search.fit(Xtrain, ytrain)

    # Get best pipeline and parameters
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best Parameters found: {best_params}")

    # ==========================================
    # 6. Evaluation with Custom Threshold
    # ==========================================
    classification_threshold = 0.4 

    # Get predicted probabilities for the positive class (class 1)
    train_probs = best_pipeline.predict_proba(Xtrain)[:, 1]
    test_probs = best_pipeline.predict_proba(Xtest)[:, 1]

    # Apply the threshold to generate class predictions
    train_preds = (train_probs >= classification_threshold).astype(int)
    test_preds = (test_probs >= classification_threshold).astype(int)

    # Generate Classification Reports (Output as dict for logging)
    train_report = classification_report(ytrain, train_preds, output_dict=True)
    test_report = classification_report(ytest, test_preds, output_dict=True)

    # ==========================================
    # 7. Logging to MLflow
    # ==========================================
    mlflow.log_params(best_params)
    mlflow.log_param("classification_threshold", classification_threshold)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # ==========================================
    # 8. Save and Register Model
    # ==========================================
    model_filename = "best_tourism_pipeline.joblib"
    joblib.dump(best_pipeline, model_filename)
    mlflow.log_artifact(model_filename, artifact_path="model")

    MODEL_REPO_ID = "dpanchali/tourism-prediction-model"
    print(f"Pushing model to Hugging Face: {MODEL_REPO_ID}...")

    try:
        try:
            api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
            print(f"Repository '{MODEL_REPO_ID}' exists.")
        except RepositoryNotFoundError:
            print(f"Creating repository '{MODEL_REPO_ID}'...")
            create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, token=HF_TOKEN)

        api.upload_file(
            path_or_fileobj=model_filename,
            path_in_repo=model_filename,
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            commit_message="Upload best XGBoost pipeline",
            token=HF_TOKEN
        )
        print("Model successfully uploaded to Hugging Face!")

    except Exception as e:
        print(f"Error during Hugging Face upload: {e}")

print("Training pipeline completed.")

# CRITICAL for GitHub Actions: 
# Keep the process alive for 10 minutes so you can click the Ngrok link.
# Without this, the runner finishes and kills the tunnel immediately.
print("Keeping process alive for 10 minutes to allow MLflow UI access via Ngrok...")
print(f"Link: {public_url}")
time.sleep(600)
