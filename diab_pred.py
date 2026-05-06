import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)
warnings.filterwarnings("ignore")
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "saved_models")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
 
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape : {df.shape}")
print(f"Target balance:\n{df['diabetes'].value_counts()}\n")
print(df.head())

FEATURES = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "hbA1c_level", "blood_glucose_level",
    "race:AfricanAmerican", "race:Asian", "race:Caucasian",
    "race:Hispanic", "race:Other"
]
TARGET = "diabetes"
 
# Keep only columns that actually exist (handles optional race columns gracefully)
FEATURES = [c for c in FEATURES if c in df.columns]
 
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train samples : {X_train.shape[0]}")
print(f"Test  samples : {X_test.shape[0]}\n")

NUMERIC_FEATURES = [
    "age", "bmi", "hbA1c_level", "blood_glucose_level"
]
CATEGORICAL_FEATURES = [
    "gender", "smoking_history"
]
BINARY_FEATURES = [
    "hypertension", "heart_disease",
    "race:AfricanAmerican", "race:Asian", "race:Caucasian",
    "race:Hispanic", "race:Other"
]

NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in FEATURES]
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c in FEATURES]
BINARY_FEATURES = [c for c in BINARY_FEATURES if c in FEATURES]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

binary_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ("bin", binary_pipeline, BINARY_FEATURES)
    ],
    remainder="drop"
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed train shape : {X_train_processed.shape}")
print(f"Processed test  shape : {X_test_processed.shape}")
print(f"Missing values before preprocessing : {int(X.isna().sum().sum())}")
print(f"Missing values after preprocessing  : {int(np.isnan(X_train_processed).sum() + np.isnan(X_test_processed).sum())}\n")

models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=50,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1
    ),
    "svm": LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000
    )
}


def get_model_scores(model, X_data):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_data)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_data)
    return model.predict(X_data)


def evaluate_model(model_name, model, X_data, y_true):
    y_pred = model.predict(X_data)
    y_score = get_model_scores(model, X_data)
    matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score)
    }

    display_name = model_name.replace("_", " ").title()
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    print(f"\n{display_name} Evaluation")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("Confusion Matrix")
    print(matrix_df)
    print()

    return metrics, y_score, matrix


def save_confusion_matrix_plot(model_name, matrix):
    display_name = model_name.replace("_", " ").title()
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"]
    )
    plt.title(f"{display_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plot_path = os.path.join(REPORT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def save_roc_curve_plot(roc_data):
    plt.figure(figsize=(7, 5))

    for model_name, y_score in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        display_name = model_name.replace("_", " ").title()
        plt.plot(fpr, tpr, label=f"{display_name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(REPORT_DIR, "roc_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


results = []
trained_models = {}
roc_data = {}
confusion_matrix_paths = {}

for model_name, model in models.items():
    print(f"Training {model_name.replace('_', ' ').title()}...")
    model.fit(X_train_processed, y_train)
    trained_models[model_name] = model

    metrics, y_score, matrix = evaluate_model(
        model_name, model, X_test_processed, y_test
    )
    results.append(metrics)
    roc_data[model_name] = y_score
    confusion_matrix_paths[model_name] = save_confusion_matrix_plot(model_name, matrix)

results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
print("Model Comparison")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]
best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")
best_pipeline_path = os.path.join(MODEL_DIR, "best_pipeline.joblib")
best_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", best_model)
])
joblib.dump(best_model, best_model_path)
joblib.dump(best_pipeline, best_pipeline_path)
print(f"\nBest model based on ROC-AUC: {best_model_name}")
print(f"Saved best model to {best_model_path}")
print(f"Saved best pipeline to {best_pipeline_path}")

results_path = os.path.join(REPORT_DIR, "model_comparison.csv")
results_df.to_csv(results_path, index=False)
roc_plot_path = save_roc_curve_plot(roc_data)

print(f"Saved model comparison to {results_path}")
print(f"Saved ROC curve plot to {roc_plot_path}")
for model_name, plot_path in confusion_matrix_paths.items():
    print(f"Saved {model_name} confusion matrix plot to {plot_path}")

preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, preprocessor_path)
print(f"\nSaved preprocessor to {preprocessor_path}")

for model_name, model in trained_models.items():
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Saved {model_name} model to {model_path}")
