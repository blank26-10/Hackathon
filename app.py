import os

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "saved_models", "best_pipeline.joblib")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
ROC_CURVE_PATH = os.path.join(BASE_DIR, "reports", "roc_curves.png")

FEATURES = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "hbA1c_level", "blood_glucose_level",
    "race:AfricanAmerican", "race:Asian", "race:Caucasian",
    "race:Hispanic", "race:Other"
]

GENDER_CODES = {
    "Female": 0.0,
    "Male": 1.0
}

SMOKING_CODES = {
    "No Info": 0,
    "Current": 1,
    "Ever": 2,
    "Former": 3,
    "Never": 4,
    "Not Current": 5
}

RACE_COLUMNS = {
    "African American": "race:AfricanAmerican",
    "Asian": "race:Asian",
    "Caucasian": "race:Caucasian",
    "Hispanic": "race:Hispanic",
    "Other": "race:Other"
}

DIET_PREFERENCES = ["Balanced", "Vegetarian", "South Indian", "Low Sodium"]

MEAL_TEMPLATES = {
    "Balanced": {
        "Breakfast": "Oats or whole-grain toast with egg/curd and a small portion of fruit",
        "Lunch": "Half plate non-starchy vegetables, grilled chicken/fish/beans, and one quarter plate brown rice/roti",
        "Snack": "Unsweetened yogurt, roasted chana, nuts, or a whole fruit",
        "Dinner": "Vegetable soup/salad with lean protein and a small whole-grain portion"
    },
    "Vegetarian": {
        "Breakfast": "Vegetable oats, besan chilla, or curd with fruit and nuts",
        "Lunch": "Half plate vegetables, dal/chana/paneer/tofu, and one quarter plate brown rice/roti",
        "Snack": "Sprouts, roasted chana, unsweetened curd, or nuts",
        "Dinner": "Dal/tofu/paneer with sauteed vegetables and a small millet/roti portion"
    },
    "South Indian": {
        "Breakfast": "2 small idlis or 1 dosa with sambar and extra vegetables; limit coconut chutney",
        "Lunch": "Vegetable poriyal, sambar/rasam, curd, and a measured rice or millet portion",
        "Snack": "Sundal, buttermilk without sugar, nuts, or a whole fruit",
        "Dinner": "Adai, vegetable upma, or chapati with dal and non-starchy vegetables"
    },
    "Low Sodium": {
        "Breakfast": "Oats, fruit, curd, or egg with no added salt seasoning",
        "Lunch": "Fresh vegetables, dal/beans/chicken/fish, and a small whole-grain portion; avoid pickles and papad",
        "Snack": "Unsalted nuts, fruit, sprouts, or unsweetened yogurt",
        "Dinner": "Home-cooked lean protein with vegetables; use lemon, herbs, and spices instead of extra salt"
    }
}


st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        max-width: 1180px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    .result-panel {
        border: 1px solid #d8dde6;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
    }
    .result-label {
        font-size: 0.85rem;
        color: #5b6472;
        margin-bottom: 0.25rem;
    }
    .result-value {
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.1;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        return None
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_model_comparison():
    if not os.path.exists(REPORT_PATH):
        return None
    return pd.read_csv(REPORT_PATH)


def build_patient_record(values):
    patient = {
        "gender": GENDER_CODES[values["gender"]],
        "age": values["age"],
        "hypertension": int(values["hypertension"]),
        "heart_disease": int(values["heart_disease"]),
        "smoking_history": SMOKING_CODES[values["smoking_history"]],
        "bmi": values["bmi"],
        "hbA1c_level": values["hba1c"],
        "blood_glucose_level": values["blood_glucose"]
    }

    for column in RACE_COLUMNS.values():
        patient[column] = 0
    patient[RACE_COLUMNS[values["race"]]] = 1

    return pd.DataFrame([patient], columns=FEATURES)


def get_probability(pipeline, patient_df):
    if hasattr(pipeline, "predict_proba"):
        return float(pipeline.predict_proba(patient_df)[0][1])
    return None


def render_prediction_result(prediction, probability):
    label = "Diabetes" if prediction == 1 else "No Diabetes"
    risk_text = "Unavailable" if probability is None else f"{probability:.1%}"
    color = "#b42318" if prediction == 1 else "#027a48"

    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-label">Prediction</div>
            <div class="result-value" style="color: {color};">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.caption('''Disclaimer: 
               The model is based on machine learning and may produce errors, especially in borderline or rare cases. It should not be used for critical healthcare decisions.''')


def get_diet_risk_level(prediction, probability, values):
    if prediction == 1 or values["hba1c"] >= 6.5 or values["blood_glucose"] >= 200:
        return "High focus"
    if (
        (probability is not None and probability >= 0.35)
        or values["hba1c"] >= 5.7
        or values["blood_glucose"] >= 140
        or values["bmi"] >= 30
    ):
        return "Moderate focus"
    return "Healthy maintenance"


def build_diet_recommendation(values, prediction, probability):
    risk_level = get_diet_risk_level(prediction, probability, values)
    preference = values["diet_preference"]
    meals = MEAL_TEMPLATES[preference]

    priorities = [
        "Use the plate method: half non-starchy vegetables, one quarter lean protein, and one quarter high-fiber carbohydrates.",
        "Choose water, unsweetened tea, or buttermilk without sugar instead of sweet drinks or fruit juice.",
        "Prefer whole foods over refined grains, sweets, packaged snacks, and deep-fried foods."
    ]

    if risk_level == "High focus":
        priorities.insert(0, "Keep carbohydrate portions consistent at each meal and avoid skipping meals.")
    if values["bmi"] >= 30:
        priorities.append("Keep cooking oil, fried snacks, sweets, and large rice/roti portions smaller to support weight control.")
    if values["hypertension"] or preference == "Low Sodium":
        priorities.append("Limit salt-heavy foods such as pickles, papad, namkeen, processed meats, instant noodles, and packaged soups.")
    if values["heart_disease"]:
        priorities.append("Favor grilled, steamed, or sauteed foods and limit ghee, butter, cream, and high-fat meats.")
    if values["blood_glucose"] >= 180 or values["hba1c"] >= 6.5:
        priorities.append("Discuss a personalized carbohydrate target with a doctor or registered dietitian.")

    return risk_level, meals, priorities


def render_diet_recommendation(values, prediction, probability):
    risk_level, meals, priorities = build_diet_recommendation(values, prediction, probability)

    st.divider()
    st.subheader("Diet Plan Recommendation")
    st.caption("General wellness guidance only. For diabetes, kidney disease, pregnancy, insulin use, or other medical needs, use a clinician or registered dietitian plan.")
    st.info(f"Plan focus: {risk_level}")

    meal_cols = st.columns(2)
    for index, (meal, suggestion) in enumerate(meals.items()):
        with meal_cols[index % 2]:
            st.markdown(f"**{meal}**")
            st.write(suggestion)

    st.markdown("**Daily priorities**")
    for item in priorities:
        st.write(f"- {item}")


st.title("Diabetes Prediction")

pipeline = load_pipeline()

if pipeline is None:
    st.error("Model file not found. Run `python diab_pred.py` first.")
    st.stop()

prediction_tab, performance_tab = st.tabs(["Prediction", "Model Performance"])

with prediction_tab:
    input_col, result_col = st.columns([1.35, 1], gap="large")

    with input_col:
        with st.form("patient_form"):
            st.subheader("Patient Details")

            row_1 = st.columns(3)
            gender = row_1[0].selectbox("Gender", list(GENDER_CODES.keys()))
            age = row_1[1].number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
            race = row_1[2].selectbox("Race", list(RACE_COLUMNS.keys()), index=2)

            row_2 = st.columns(3)
            bmi = row_2[0].number_input("BMI", min_value=10.0, max_value=80.0, value=27.5, step=0.1)
            hba1c = row_2[1].number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.7, step=0.1)
            blood_glucose = row_2[2].number_input(
                "Blood Glucose Level",
                min_value=50,
                max_value=350,
                value=140,
                step=1
            )

            row_3 = st.columns(3)
            smoking_history = row_3[0].selectbox("Smoking History", list(SMOKING_CODES.keys()), index=4)
            hypertension = row_3[1].checkbox("Hypertension")
            heart_disease = row_3[2].checkbox("Heart Disease")

            diet_preference = st.selectbox("Diet Preference", DIET_PREFERENCES)

            submitted = st.form_submit_button("Predict")

    with result_col:
        st.subheader("Result")

        if submitted:
            form_values = {
                "gender": gender,
                "age": age,
                "race": race,
                "bmi": bmi,
                "hba1c": hba1c,
                "blood_glucose": blood_glucose,
                "smoking_history": smoking_history,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "diet_preference": diet_preference
            }
            patient_df = build_patient_record(form_values)
            prediction = int(pipeline.predict(patient_df)[0])
            probability = get_probability(pipeline, patient_df)
            render_prediction_result(prediction, probability)
            render_diet_recommendation(form_values, prediction, probability)
        else:
            st.info("Enter patient details and click Predict.")

with performance_tab:
    comparison_df = load_model_comparison()

    st.subheader("Model Comparison")
    if comparison_df is not None:
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Model comparison report not found.")

    if os.path.exists(ROC_CURVE_PATH):
        st.image(ROC_CURVE_PATH, caption="ROC curves", use_container_width=True)

    matrix_cols = st.columns(3)
    for index, model_name in enumerate(["logistic_regression", "random_forest", "svm"]):
        matrix_path = os.path.join(BASE_DIR, "reports", f"{model_name}_confusion_matrix.png")
        if os.path.exists(matrix_path):
            title = model_name.replace("_", " ").title()
            matrix_cols[index].image(matrix_path, caption=title, use_container_width=True)
