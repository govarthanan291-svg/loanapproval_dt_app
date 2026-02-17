import streamlit as st
import pickle
import numpy as np

# ---------- Page Config ----------
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# ---------- Background + Global Style ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #ff512f, #dd2476);
    color: white;
}
.section-box {
    background-color: rgba(0,0,0,0.4);
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 20px;
}
.stButton>button {
    height: 50px;
    width: 200px;
    font-size: 18px;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- BIG TOP TITLE BOX ----------
st.markdown("""
<div style="
    background: linear-gradient(to right, #FFD700, #FF8C00);
    padding:40px;
    border-radius:25px;
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:black;
    box-shadow:0 0 30px #FFA500;
    margin-bottom:30px;">
    🏦 LOAN APPROVAL PREDICTION SYSTEM
</div>
""", unsafe_allow_html=True)

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
    predict = st.button("🔮 Predict")

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

    if prediction[0] == 1:
        st.markdown("""
        <div style="
            background-color:#00C853;
            padding:50px;
            border-radius:25px;
            text-align:center;
            font-size:45px;
            font-weight:bold;
            color:white;
            box-shadow:0 0 30px #00FF00;
            margin-top:30px;">
            🎉 LOAN APPROVED 🎉
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            background-color:#D50000;
            padding:50px;
            border-radius:25px;
            text-align:center;
            font-size:45px;
            font-weight:bold;
            color:white;
            box-shadow:0 0 30px red;
            margin-top:30px;">
            ❌ LOAN REJECTED ❌
        </div>
        """, unsafe_allow_html=True)

# ---------- Clear ----------
if clear:
    st.rerun()
