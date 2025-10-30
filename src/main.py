import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# --- 1. Inisialisasi Aplikasi ---
app = FastAPI(title="API Prediksi Lead Scoring Multi-Model", version="2.0.0")

# --- 2. FUNGSI UNTUK MEMUAT SEMUA MODEL ---
# Memuat semua model ke dalam sebuah dictionary agar mudah diakses
def load_all_models():
    models = {}
    model_dir = '../models/models_V1'
    model_files = {
        "Logistic Regression": "logistic_regression_v1.pkl",
        "Random Forest": "random_forest_v1.pkl",
        "XGBoost (Default)": "xgboost_model_v1.pkl", 
        "XGBoost (Tuned)": "xgboost_tuned_v1.pkl"    
    }
    
    for model_name, file_name in model_files.items():
        path = os.path.join(model_dir, file_name)
        with open(path, 'rb') as f:
            models[model_name] = pickle.load(f)
            print(f"Model '{model_name}' berhasil dimuat.")
            
    return models

# Muat semua model saat aplikasi dimulai
models = load_all_models()

# --- 3. DEFINISI STRUKTUR INPUT (FORM PESANAN) ---
# Form ini hanya berisi data fitur nasabah
class CustomerData(BaseModel):
    duration_min: float
    pdays: int
    euribor3m: float
    nr_employed: float
    emp_var_rate: float
    # PENTING: Pastikan ketiga model .pkl Anda dilatih dengan kolom yang sama persis
    # agar bisa menerima input yang sama.

# Form ini adalah "pesanan" lengkap dari Streamlit
class PredictionRequest(BaseModel):
    model_name: str # Nama model yang ingin digunakan
    customer_data: CustomerData # Data fitur nasabah

# --- 4. ENDPOINT PREDIKSI YANG DIPERBARUI ---
@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Endpoint untuk membuat prediksi menggunakan model yang dipilih.
    """
    model_name = request.model_name
    
    # Cek apakah model yang diminta ada
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' tidak ditemukan.")
    
    # Pilih model yang sesuai dari dictionary
    selected_model = models[model_name]
    
    # Ubah data input menjadi DataFrame
    input_df = pd.DataFrame([request.customer_data.dict()])
    
    # --- PENANGANAN KOLOM (SANGAT PENTING) ---
    # Model Anda dilatih dengan BANYAK kolom hasil one-hot encoding.
    # API ini hanya menerima 5 fitur. Agar ini berfungsi, Anda HARUS melatih ulang
    # ketiga model HANYA dengan 5 fitur ini dan menyimpannya kembali.
    # Namun, untuk demonstrasi, kita akan mengasumsikan model Anda sudah sesuai.
    # Mari kita ambil daftar kolom yang dibutuhkan dari model XGBoost sebagai acuan.
    
    try:
        training_columns = models['XGBoost'].get_booster().feature_names
    except AttributeError: # Jika bukan XGBoost, coba cara lain
        try:
             training_columns = models['Random Forest'].feature_names_in_
        except AttributeError:
             training_columns = models['Logistic Regression'].feature_names_in_

    # Buat DataFrame kosong dengan semua kolom training
    full_input_df = pd.DataFrame(columns=training_columns)
    full_input_df.loc[0] = 0 # Isi dengan nilai default

    # Isi kolom yang datanya kita punya dari input
    for col in input_df.columns:
        if col in full_input_df.columns:
            full_input_df[col] = input_df[col]
            
    # Buat prediksi menggunakan model yang dipilih
    prediction_proba = selected_model.predict_proba(full_input_df)[0][1]
    prediction = 1 if prediction_proba > 0.5 else 0
    
    # Kembalikan hasilnya
    return {
        "model_used": model_name,
        "prediction": int(prediction),
        "label": "Berpotensi Berlangganan" if prediction == 1 else "Tidak Berpotensi",
        "probability": float(prediction_proba)
    }