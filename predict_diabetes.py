import os

import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "saved_models", "best_pipeline.joblib")

FEATURES = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "hbA1c_level", "blood_glucose_level",
    "race:AfricanAmerican", "race:Asian", "race:Caucasian",
    "race:Hispanic", "race:Other"
]


def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            "best_pipeline.joblib was not found. Run diab_pred.py first."
        )
    return joblib.load(PIPELINE_PATH)


def predict_diabetes(patient_data):
    pipeline = load_pipeline()
    patient_df = pd.DataFrame([patient_data], columns=FEATURES)
    prediction = int(pipeline.predict(patient_df)[0])

    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        probability = float(pipeline.predict_proba(patient_df)[0][1])
    else:
        probability = None

    return {
        "prediction": prediction,
        "label": "Diabetes" if prediction == 1 else "No Diabetes",
        "diabetes_probability": probability
    }


if __name__ == "__main__":
    sample_patient = {
        "gender": 1.0,
        "age": 55.0,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": 4,
        "bmi": 31.2,
        "hbA1c_level": 6.1,
        "blood_glucose_level": 180,
        "race:AfricanAmerican": 0,
        "race:Asian": 0,
        "race:Caucasian": 1,
        "race:Hispanic": 0,
        "race:Other": 0
    }

    result = predict_diabetes(sample_patient)
    print("Prediction Result")
    print(f"Label       : {result['label']}")
    print(f"Prediction  : {result['prediction']}")

    if result["diabetes_probability"] is not None:
        print(f"Probability : {result['diabetes_probability']:.4f}")
