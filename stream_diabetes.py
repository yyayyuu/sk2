import streamlit as st
import pandas as pd
import pickle

# =========================
# Load model dan kolom
# =========================
model = pickle.load(open("model_xgb_tomek.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("Prediksi Diabetes (XGBoost + TomekLinks)")

# =========================
# Input form
# =========================
age = st.number_input("Umur", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=32.0)
blood_glucose_level = st.number_input("Blood Glucose (mg/dL)", min_value=0, max_value=500, value=180)
hbA1c_level = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=8.0)
smoking_history = st.selectbox("Riwayat Merokok", ["never", "current", "former", "ever"])
hypertension = st.selectbox("Hipertensi", [0, 1])
heart_disease = st.selectbox("Penyakit Jantung", [0, 1])

# =========================
# Tombol prediksi
# =========================
if st.button("Prediksi"):
    data_input = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "blood_glucose_level": [blood_glucose_level],
        "hbA1c_level": [hbA1c_level],
        "smoking_history": [smoking_history],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease]
    })
    
    # One-Hot Encoding kolom kategorikal
    categorical_cols = ["gender", "smoking_history"]
    for col in categorical_cols:
        data_input[col] = data_input[col].astype(str)
    
    data_input_encoded = pd.get_dummies(data_input)
    
    # Tambahkan kolom yang hilang agar sama dengan training
    for c in model_columns:
        if c not in data_input_encoded.columns:
            data_input_encoded[c] = 0
    
    # Urutkan kolom sesuai training
    data_input_encoded = data_input_encoded[model_columns]
    
    # Prediksi
    prediksi = model.predict(data_input_encoded)
    st.success(f"Hasil prediksi diabetes: {'Diabetes' if prediksi[0]==1 else 'Tidak Diabetes'}")
