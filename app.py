import streamlit as st
import pandas as pd
import pickle
import os

# --- 1. Konfigurasi Halaman ---
st.set_page_config(page_title="Dasbor Lead Scoring V2", page_icon="📊", layout="wide")

# --- 2. Judul dan Deskripsi ---
st.title("📊 Dasbor Analisis Model Lead Scoring (Versi Validated V2)")
st.write(
    "Dasbor ini mendemonstrasikan dan membandingkan kinerja model-model Machine Learning V2 yang valid (bebas data leakage) "
    "untuk memprediksi potensi nasabah berlangganan deposito berjangka."
)
st.markdown("---")

# --- 3. FUNGSI UNTUK MEMUAT MODEL V2 ---
@st.cache_resource
def load_models_v2():
    """Memuat semua model V2 yang valid dari folder yang benar."""
    models = {}
    model_dir = os.path.join('models', 'models_V2')
    model_files = {
        "Logistic Regression V2": "logistic_regression_v2.pkl",
        "Random Forest V2": "random_forest_v2.pkl",
        "XGBoost (Default) V2": "xgboost_default_v2.pkl", 
        "XGBoost (Tuned) V2": "xgboost_tuned_v2.pkl"    
    }
    for name, file in model_files.items():
        path = os.path.join(model_dir, file)
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    return models

MODELS = load_models_v2()

# --- Ambil daftar kolom training sebagai referensi ---
try:
    TRAINING_COLUMNS = MODELS["XGBoost (Tuned) V2"].get_booster().feature_names
except:
    TRAINING_COLUMNS = MODELS["Random Forest V2"].feature_names_in_

# --- 4. DATA KINERJA MODEL V2 (DARI NOTEBOOK 02B-Modeling.ipynb) ---
model_performance_v2 = {
    "Logistic Regression V2": {"Accuracy": 0.8309, "Precision": 0.3602, "Recall": 0.6455, "F1-Score": 0.4624},
    "Random Forest V2":     {"Accuracy": 0.8935, "Precision": 0.5537, "Recall": 0.2834, "F1-Score": 0.3749},
    "XGBoost (Default) V2": {"Accuracy": 0.8451, "Precision": 0.3816, "Recall": 0.6045, "F1-Score": 0.4679},
    "XGBoost (Tuned) V2":   {"Accuracy": 0.8537, "Precision": 0.4064, "Recall": 0.6476, "F1-Score": 0.4994} 
}

# --- 5. SIDEBAR UNTUK INPUT DATA MENTAH DARI PENGGUNA ---
st.sidebar.header("⚙️ Pengaturan & Input Data Nasabah")
model_selection = st.sidebar.selectbox("Pilih Model untuk Prediksi:", list(MODELS.keys()))
st.sidebar.markdown("---")

def get_user_input_features():
    """Mengumpulkan data mentah dari pengguna melalui widget Streamlit."""
    st.sidebar.subheader("Data Demografi & Finansial")
    age = st.sidebar.slider("Usia", 18, 98, 40)
    job_options = ['admin.','blue-collar','technician','services','management','retired','entrepreneur','self-employed','housemaid','unemployed','student']
    job = st.sidebar.selectbox("Pekerjaan", job_options, index=0)
    marital_options = ['married','single','divorced']
    marital = st.sidebar.selectbox("Status Pernikahan", marital_options, index=0)
    education_options = ['university.degree','high.school','basic.9y','professional.course','basic.4y','basic.6y','illiterate']
    education = st.sidebar.selectbox("Pendidikan", education_options, index=0)

    st.sidebar.subheader("Riwayat Kampanye & Kontak")
    pdays = st.sidebar.number_input("Hari sejak kontak terakhir (masukkan 999 jika belum pernah)", min_value=0, max_value=999, value=999)
    previous = st.sidebar.number_input("Jumlah kontak sebelumnya", min_value=0, max_value=10, value=0)
    poutcome_options = ['nonexistent','failure','success']
    poutcome = st.sidebar.selectbox("Hasil kampanye sebelumnya", poutcome_options, index=0)
    contact_options = ['cellular', 'telephone']
    contact = st.sidebar.selectbox("Metode Kontak", contact_options, index=0)
    
    st.sidebar.subheader("Indikator Ekonomi")
    euribor3m = st.sidebar.slider("Suku Bunga Euribor 3 Bulan", 0.5, 5.5, 4.857, 0.001)
    nr_employed = st.sidebar.slider("Jumlah Karyawan (ribuan)", 4900.0, 5300.0, 5191.0, 0.1)

    # Buat dictionary dari input
    data = {
        'age': age, 'job': job, 'marital': marital, 'education': education, 'default': 'no', 
        'housing': 'no', 'loan': 'no', 'contact': contact, 'month': 'may', 'day_of_week': 'fri',
        'campaign': 1, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp_var_rate': 1.1, 'cons_price_idx': 93.994, 'cons_conf_idx': -36.4, 
        'euribor3m': euribor3m, 'nr_employed': nr_employed
    }
    return pd.DataFrame(data, index=[0])

raw_df = get_user_input_features()

# --- 6. FUNGSI PREPROCESSING ---
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Mereplikasi pipeline preprocessing dari notebook untuk data input."""
    df['pernah_dihubungi'] = (df['pdays'] != 999).astype(int)
    df = df.drop('pdays', axis=1)
    
    modes = {
        'job': 'admin.', 'marital': 'married', 'education': 'university.degree',
        'default': 'no', 'housing': 'yes', 'loan': 'no'
    }
    for col, mode_val in modes.items():
        df[col] = df[col].replace('unknown', mode_val)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    df_final = df_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    return df_final

processed_df = preprocess_input(raw_df.copy())

# --- 7. TAMPILAN UTAMA ---
col1, col2 = st.columns([0.6, 0.4]) # Beri lebih banyak ruang untuk kolom kiri

with col1:
    st.subheader(f"Kinerja Model: {model_selection}")
    perf = model_performance_v2[model_selection]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{perf['Accuracy']:.4f}")
    c2.metric("Precision (Yes)", f"{perf['Precision']:.4f}")
    c3.metric("Recall (Yes)", f"{perf['Recall']:.4f}")
    c4.metric("F1-Score (Yes)", f"{perf['F1-Score']:.4f}")

    st.markdown("---")
    st.subheader("Hasil Prediksi untuk Nasabah Ini:")
    
    selected_model = MODELS[model_selection]
    prediction_proba = selected_model.predict_proba(processed_df)[0][1]
    prediction = 1 if prediction_proba > 0.5 else 0
    
    if prediction == 1:
        st.success(f"**BERPOTENSI BERLANGGANAN**")
    else:
        st.error(f"**TIDAK BERPOTENSI**")
    
    # Ubah probabilitas menjadi integer persentase untuk progress bar
    progress_value = int(prediction_proba * 100)
    st.progress(progress_value)
    
    st.write(f"**Probabilitas untuk Berlangganan: {prediction_proba*100:.2f}%**")
    
    st.markdown("---")
    st.subheader("Analisis Singkat Model yang Dipilih")
    if "XGBoost (Tuned)" in model_selection:
        st.info("Anda memilih **XGBoost (Tuned) V2**, model terbaik. Model ini memberikan keseimbangan optimal (F1-Score tertinggi) antara menemukan prospek potensial (Recall) dan efisiensi tim penjualan (Precision).")
    elif "Logistic Regression" in model_selection:
        st.warning("Anda memilih **Logistic Regression V2**. Model ini baik dalam menjangkau banyak prospek (Recall cukup tinggi), namun dengan risiko banyak prediksi positif yang salah (Precision rendah).")
    elif "Random Forest" in model_selection:
        st.error("Anda memilih **Random Forest V2**. Model ini sangat 'hati-hati' dan cenderung memprediksi 'Tidak'. Akibatnya, ia sangat sering melewatkan prospek yang sebenarnya potensial (Recall sangat rendah).")


with col2:
    st.subheader("Data Nasabah yang Dimasukkan (Mentah)")
    # Tampilkan hanya kolom yang relevan dari input pengguna
    display_cols = ['age', 'job', 'marital', 'education', 'pdays', 'previous', 'poutcome', 'contact', 'euribor3m', 'nr_employed']
    st.dataframe(raw_df[display_cols].T.rename(columns={0: 'Nilai'}))