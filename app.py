import streamlit as st
import pickle
import numpy as np

# ---------- Page Config ----------
st.set_page_config(page_title="Loan Prediction App", layout="centered")

# ---------- Dark Orange Background ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FF8C00;
        color: white;
    }
    .stButton>button {
        background-color: black;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Loan Approval Prediction App")

# ---------- Load Pickle Files ----------
model = pickle.load(open("decision_tree_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ---------- User Inputs ----------
age = st.number_input("Enter Age")
income = st.number_input("Enter Income")
credit_score = st.number_input("Enter Credit Score")
loan_amount = st.number_input("Enter Loan Amount")

employment_type = st.selectbox(
    "Employment Type",
    ["Salaried", "Self-Employed", "Business"]
)

dependents = st.number_input("Number of Dependents")

# ---------- Predict Button ----------
if st.button("Predict Loan Status"):

    # Encode Employment Type
    employment_encoded = encoder.transform([employment_type])[0]

    # Convert input into array
    input_data = np.array([[age, income, credit_score,
                            loan_amount, employment_encoded, dependents]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # PCA Transform
    input_pca = pca.transform(input_scaled)

    # Prediction
    prediction = model.predict(input_pca)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
