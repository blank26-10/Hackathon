# Diabetes Risk Prediction

> *Early-stage diabetes risk classification using clinical health indicators — deployed as a live web application.*

[![Live App](https://img.shields.io/badge/%20Live%20App-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://hackathon-c.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn)](https://scikit-learn.org)

---

## Problem Statement

Diabetes affects over 77 million people in India alone, and a large fraction of cases go undiagnosed until serious complications emerge. Early detection using routinely available clinical data can change outcomes dramatically. This project builds a binary classification model to predict diabetes risk from health survey indicators, optimized to **minimize missed cases** — because in healthcare, a false negative (missing a true diabetic) is far more costly than a false positive.

---

## Pipeline Overview

```
Raw Clinical Data
    │
    ▼
Data Preprocessing
    ├── Missing value handling
    ├── Feature scaling (StandardScaler)
    └── Class imbalance treatment (balanced class weights)
    │
    ▼
Model Training
    └── Random Forest Classifier
    │
    ▼
Threshold Optimization
    └── Lowered classification threshold → improved recall
    │
    ▼
Evaluation
    ├── ROC-AUC Score
    ├── Precision / Recall / F1
    └── Confusion Matrix
    │
    ▼
Streamlit Web App → hackathon-c.streamlit.app
```

---

## Dataset

| Feature | Description |
|---------|-------------|
| **Size** | ~100,000 rows |
| **Target** | Binary: Diabetic / Non-diabetic |
| **Features** | Clinical indicators (BMI, age, blood pressure, glucose, etc.) |
| **Challenge** | Class imbalance — diabetic cases are a minority class |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.9+ |
| **ML** | scikit-learn · Random Forest |
| **Data** | pandas · NumPy |
| **Visualization** | matplotlib · seaborn |
| **Deployment** | Streamlit |

---

## Key Design Decisions

**Handling Class Imbalance**  
The dataset has significantly more non-diabetic cases than diabetic ones. Using `class_weight='balanced'` in Random Forest adjusts the model to treat minority class errors as more costly — pushing it toward better recall on diabetic cases.

**Threshold Optimization**  
By default, a classifier predicts "positive" when probability ≥ 0.5. Lowering this threshold (e.g., to 0.35–0.40) increases sensitivity — the model flags more people as potentially diabetic, at the cost of slightly more false positives. For a screening tool, this tradeoff is the right one.

**Why Random Forest?**  
Random Forest handles mixed feature types well, is robust to outliers, and provides feature importances — making it interpretable enough for healthcare contexts where understanding *why* a prediction is made matters.

---

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | *0.95801* |
| Precision | *0.88* |
| F1 Score | *0.80* |

---

## Getting Started

**[hackathon-c.streamlit.app](https://hackathon-c.streamlit.app)**

---

## Repository Structure

```
diabetes-prediction/
├── app.py                # Streamlit web application
├── requirements.txt      # Dependencies
├── diab_pred.py          # EDA and model training
├── models/               # Saved model (.pkl)
└── data/                 # Dataset files
```

---

