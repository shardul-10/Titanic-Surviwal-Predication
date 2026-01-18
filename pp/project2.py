import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="centered"
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
try:
    model = joblib.load("logistic_model.pkl")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🚢 Titanic Survival Prediction")

st.sidebar.header("Passenger Details")

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
pclass = st.sidebar.slider("Passenger Class", 1, 3, 1)
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.slider("Siblings / Spouse", 0, 8, 0)
parch = st.sidebar.slider("Parents / Children", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0, 500, 50)

sex_input = st.sidebar.selectbox("Gender", ["Male", "Female"])
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

# --------------------------------------------------
# Encoding (MUST MATCH TRAINING)
# --------------------------------------------------

# Sex encoding: Male = 0, Female = 1
sex = 0 if sex_input == "Male" else 1

# Embarked one-hot encoding (NO DROP)
embarked_dict = {"C": 0, "Q": 0, "S": 0}
embarked_dict[embarked] = 1

# --------------------------------------------------
# Create Input DataFrame
# --------------------------------------------------
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "C": embarked_dict["C"],
    "Q": embarked_dict["Q"],
    "S": embarked_dict["S"]
}])

# --------------------------------------------------
# Force correct feature order
# --------------------------------------------------
FEATURE_ORDER = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "C",
    "Q",
    "S"
]

input_data = input_data[FEATURE_ORDER]

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.write(f"### Survival Probability: {probability[0][1]:.2f}")

        if prediction[0] == 1:
            st.success("✅ Passenger Survived")
        else:
            st.error("❌ Passenger Did Not Survive")

    except Exception as e:
        st.error(f"Prediction error: {e}")