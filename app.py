import streamlit as st
import pickle
import numpy as np

# ---------- Page Config ----------
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #ff512f, #dd2476);
    color: white;
}

.big-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #FFD700;
}

.section-box {
    background-color: rgba(0,0,0,0.4);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.stButton>button {
    background-color: #00FFFF;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
    width: 200px;
}

.predict-btn>button {
    background-color: #32CD32;
    color: black;
}

.clear-btn>button {
    background-color: red;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<p class="big-title">🚀 Loan Approval Prediction System</p>', unsafe_allow_html=True)

# ---------- Load Pickle Files ----------
model = pickle.load(open("decision_tree_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ---------- Layout ----------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("👤 Personal Details")

    age = st.number_input("Age", min_value=18, max_value=100)
    dependents = st.number_input("Dependents", min_value=0)

    employment_type = st.selectbox(
        "Employment Type",
        ["Salaried", "Self-Employed", "Business"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("💰 Financial Details")

    income = st.number_input("Income")
    credit_score = st.number_input("Credit Score")
    loan_amount = st.number_input("Loan Amount")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Buttons ----------
col3, col4 = st.columns(2)

with col3:
    predict = st.button("🔮 Predict Loan Status")

with col4:
    clear = st.button("🗑 Clear")

# ---------- Prediction ----------
if predict:
    employment_encoded = encoder.transform([employment_type])[0]

    input_data = np.array([[age, income, credit_score,
                            loan_amount, employment_encoded, dependents]])

    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)

    prediction = model.predict(input_pca)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    if prediction[0] == 1:
        st.success("🎉 Loan Approved Successfully!")
    else:
        st.error("❌ Loan Rejected!")

    st.markdown('</div>', unsafe_allow_html=True)

if clear:
    st.rerun()
