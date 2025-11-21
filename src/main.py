import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np

# --- 1. Inisialisasi Aplikasi ---
app = FastAPI(title="API Prediksi Lead Scoring V2 (VALIDATED)", version="2.0.0")

# --- 2. FUNGSI UNTUK MEMUAT SEMUA MODEL V2 ---
def load_all_models_v2():
    """Memuat semua model V2 yang valid ke dalam dictionary."""
    models = {}
    model_dir = os.path.join('..', 'models', 'models_V2') # Path yang benar
    
    # Pastikan nama file ini sesuai dengan yang ada di folder models_V2 Anda
    model_files = {
        "Logistic Regression V2": "logistic_regression_v2.pkl",
        "Random Forest V2": "random_forest_v2.pkl",
        "XGBoost (Default) V2": "xgboost_default_v2.pkl", 
        "XGBoost (Tuned) V2": "xgboost_tuned_v2.pkl"    
    }
    
    for model_name, file_name in model_files.items():
        path = os.path.join(model_dir, file_name)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)
                print(f"Model '{model_name}' berhasil dimuat.")
        else:
            print(f"PERINGATAN: File model tidak ditemukan di '{path}'")
            
    return models

# Muat semua model V2 saat aplikasi dimulai
MODELS = load_all_models_v2()

# --- Ambil daftar kolom training dari model final sebagai referensi ---
# Ini sangat penting untuk memastikan input memiliki format yang benar setelah encoding
try:
    TRAINING_COLUMNS = MODELS["XGBoost (Tuned) V2"].get_booster().feature_names
except Exception:
    # Fallback jika model XGBoost tidak ada, gunakan model lain
    TRAINING_COLUMNS = MODELS["Random Forest V2"].feature_names_in_

# --- 3. DEFINISI STRUKTUR INPUT (DATA MENTAH) ---
# Struktur ini mencerminkan data mentah yang akan diinput oleh pengguna, BUKAN data yang sudah diproses.
class RawCustomerData(BaseModel):
    age: int = 40
    job: str = "admin."
    marital: str = "married"
    education: str = "university.degree"
    default: str = "no"
    housing: str = "yes"
    loan: str = "no"
    contact: str = "cellular"
    month: str = "may"
    day_of_week: str = "fri"
    campaign: int = 1
    pdays: int = 999 # 999 berarti belum pernah dihubungi
    previous: int = 0
    poutcome: str = "nonexistent"
    emp_var_rate: float = 1.1
    cons_price_idx: float = 93.994
    cons_conf_idx: float = -36.4
    euribor3m: float = 4.857
    nr_employed: float = 5191.0

# Form "pesanan" lengkap
class PredictionRequest(BaseModel):
    model_name: str
    customer_data: RawCustomerData

# --- 4. PIPELINE PREPROCESSING ---
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fungsi ini mereplikasi pipeline preprocessing dari notebook 01B-EDA.ipynb
    untuk mengubah data mentah menjadi format yang siap untuk model.
    """
    # Perbaikan 1: 'duration' sudah tidak ada di input, jadi tidak perlu dihapus.
    
    # Perbaikan 2: Penanganan 'pdays' yang Benar
    df['pernah_dihubungi'] = (df['pdays'] != 999).astype(int)
    df = df.drop('pdays', axis=1)
    
    # Perbaikan 3: Imputasi 'unknown' (menggunakan modus dari data training)
    # Catatan: Di lingkungan produksi, nilai-nilai ini seharusnya disimpan saat training
    # atau diambil dari sebuah 'preprocessor' object. Di sini kita hardcode untuk simplicity.
    modes = {
        'job': 'admin.', 'marital': 'married', 'education': 'university.degree',
        'default': 'no', 'housing': 'yes', 'loan': 'no'
    }
    for col, mode_val in modes.items():
        df[col] = df[col].replace('unknown', mode_val)

    # Perbaikan 4: One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Finalisasi: Pastikan semua kolom training ada dan dalam urutan yang benar
    # Kolom yang tidak ada di input akan diisi dengan 0.
    df_final = df_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    
    return df_final

# --- 5. ENDPOINT PREDIKSI V2 ---
@app.get("/")
def read_root():
    return {"status": "API Prediksi Lead Scoring V2 sedang berjalan."}

@app.post("/predict_v2")
def predict(request: PredictionRequest):
    """
    Endpoint untuk membuat prediksi menggunakan model V2 yang valid dan pipeline preprocessing.
    """
    model_name = request.model_name
    
    # Cek apakah model V2 yang diminta ada
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' tidak ditemukan. Model yang tersedia: {list(MODELS.keys())}")
    
    # Pilih model yang sesuai
    selected_model = MODELS[model_name]
    
    # 1. Ubah data input menjadi DataFrame
    input_df = pd.DataFrame([request.customer_data.dict()])
    
    # 2. Lakukan preprocessing pada data input
    processed_df = preprocess_input(input_df)
    
    # 3. Buat prediksi
    try:
        prediction_proba = selected_model.predict_proba(processed_df)[0][1]
        prediction = 1 if prediction_proba > 0.5 else 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi error saat prediksi: {e}")
    
    # 4. Kembalikan hasilnya
    return {
        "model_used": model_name,
        "prediction": prediction,
        "label": "Berpotensi Berlangganan" if prediction == 1 else "Tidak Berpotensi",
        "probability_of_subscription": float(prediction_proba)
    }