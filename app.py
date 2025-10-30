import streamlit as st
import pandas as pd
import pickle
import os

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Dasbor Analisis Model Lead Scoring", page_icon="📊", layout="wide")

# --- Judul dan Deskripsi ---
st.title("📊 Dasbor Analisis Model untuk Prediksi Lead Scoring")
st.write(
    "Aplikasi ini mendemonstrasikan dan membandingkan kinerja tiga model Machine Learning "
    "untuk memprediksi potensi nasabah berlangganan deposito berjangka."
)

# --- FUNGSI UNTUK MEMUAT MODEL ---
@st.cache_resource
def load_models():
    models = {}
    models['Logistic Regression'] = pickle.load(open('models/logistic_regression_v1.pkl', 'rb'))
    models['Random Forest'] = pickle.load(open('models/random_forest_v1.pkl', 'rb'))
    models['XGBoost (Default)'] = pickle.load(open('models/xgboost_model_v1.pkl', 'rb'))
    models['XGBoost (Tuned)'] = pickle.load(open('models/xgboost_tuned_v1.pkl', 'rb')) 
    return models

models = load_models()

# --- DATA KINERJA MODEL (DARI NOTEBOOK 02-Modeling.ipynb) ---
# *** PERBARUI BAGIAN INI DENGAN NILAI AKURASI BARU ***
model_performance = {
    "Logistic Regression": {"Accuracy": 0.864, "Precision": 0.45, "Recall": 0.91, "F1-Score": 0.60},
    "Random Forest": {"Accuracy": 0.914, "Precision": 0.69, "Recall": 0.44, "F1-Score": 0.54},
    "XGBoost (Default)": {"Accuracy": 0.889, "Precision": 0.50, "Recall": 0.87, "F1-Score": 0.63},
    "XGBoost (Tuned)": {"Accuracy": 0.889, "Precision": 0.50, "Recall": 0.88, "F1-Score": 0.64} 
}
# Pastikan Anda mengganti nilai di atas dengan hasil akurat dari notebook Anda

# --- SIDEBAR UNTUK INPUT PENGGUNA ---
st.sidebar.header("⚙️ Pengaturan & Input Data")
model_selection = st.sidebar.selectbox("Pilih Model untuk Prediksi:", list(models.keys()))
st.sidebar.markdown("---")
st.sidebar.header("Masukkan Data Nasabah:")

def user_input_features():
    duration_min = st.sidebar.slider("Durasi Panggilan (menit)", 0.0, 60.0, 5.0, 0.1)
    pdays = st.sidebar.slider("Hari sejak kontak terakhir (0 jika belum pernah)", 0, 30, 0)
    euribor3m = st.sidebar.slider("Suku Bunga Euribor 3 Bulan", 0.5, 5.5, 1.5, 0.1)
    nr_employed = st.sidebar.slider("Jumlah Karyawan (ribuan)", 4900.0, 5300.0, 5100.0, 1.0)
    emp_var_rate = st.sidebar.slider("Tingkat Variasi Ketenagakerjaan", -4.0, 2.0, -1.8, 0.1)
    
    training_columns = models['XGBoost'].get_booster().feature_names
    input_df = pd.DataFrame(columns=training_columns)
    input_df.loc[0] = 0 
    
    input_df['duration_min'] = duration_min
    input_df['pdays'] = pdays
    input_df['euribor3m'] = euribor3m
    input_df['nr_employed'] = nr_employed
    input_df['emp_var_rate'] = emp_var_rate
    
    return input_df

input_df = user_input_features()

# --- TAMPILAN UTAMA ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Kinerja Model: {model_selection}")
    
    perf = model_performance[model_selection]
    
    # *** PERBARUI BAGIAN INI UNTUK MENAMPILKAN 4 METRIK ***
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{perf['Accuracy']:.3f}")
    c2.metric("Precision", f"{perf['Precision']:.2f}")
    c3.metric("Recall", f"{perf['Recall']:.2f}")
    c4.metric("F1-Score", f"{perf['F1-Score']:.2f}")

    st.markdown("---")
    st.subheader("Prediksi untuk Nasabah Ini:")
    
    selected_model = models[model_selection]
    
    try:
        prediction_proba = selected_model.predict_proba(input_df)[0][1]
        prediction = 1 if prediction_proba > 0.5 else 0
        
        if prediction == 1:
            st.success(f"**Berpotensi Berlangganan**")
        else:
            st.error(f"**Tidak Berpotensi**")
        
        st.progress(prediction_proba)
        st.write(f"**Probabilitas untuk Berlangganan: {prediction_proba*100:.2f}%**")
        
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

with col2:
    st.subheader("Data Nasabah yang Dimasukkan:")
    st.dataframe(input_df[['duration_min', 'pdays', 'euribor3m', 'nr_employed', 'emp_var_rate']].T.rename(columns={0: 'Nilai'}))
    st.markdown("---")
    
    st.subheader("Ringkasan Analisis Model")
    if model_selection == "XGBoost":
        st.info("Anda memilih XGBoost, model dengan F1-Score tertinggi. Model ini memberikan keseimbangan terbaik antara menemukan prospek (Recall) dan tidak membuang waktu tim penjualan (Precision).")
    elif model_selection == "Logistic Regression":
        st.warning("Anda memilih Logistic Regression. Model ini sangat baik dalam menemukan hampir semua prospek (Recall tertinggi), tetapi menghasilkan banyak prediksi positif palsu (Precision terendah).")
    else: # Random Forest
        st.warning("Anda memilih Random Forest. Model ini sangat 'hati-hati' dan akurat saat memprediksi 'Ya' (Precision tertinggi), tetapi melewatkan banyak sekali prospek potensial (Recall terendah).")