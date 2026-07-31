import streamlit as st
import pandas as pd
import joblib

# 1. Load "Paket Otak" yang kamu buat
paket_otak = joblib.load('credit_risk_model_CART.pkl')

# Bongkar isi paketnya
model = paket_otak['model']
scaler = paket_otak['scaler']
list_fitur = paket_otak['list_fitur']
# (Kita tidak wajib pakai imputer di sini karena input Streamlit biasanya tidak membiarkan nilai kosong)

# 2. Bikin Judul di Website
st.title("Dashboard Prediksi Kelayakan Debitur")
st.write("Masukkan data nasabah baru untuk melihat prediksi kelayakan pembiayaan.")
st.markdown("---")

# 3. Bikin Kotak Isian (Form Input)
# CONTOH: Sesuaikan variabel di bawah ini dengan fitur yang ada di modelmu
col1, col2 = st.columns(2)

with col1:
    plafond_fix = st.number_input("Nilai Plafond", min_value=0.0, format="%.4f")
    tenor_fix = st.number_input("Jangka Waktu / Tenor", min_value=0.0, format="%.4f")

with col2:
    resiko_usaha = st.selectbox("Risiko Usaha", ["Rendah", "Sedang", "Tinggi"])
    # Konversi jawaban jadi angka sesuai format datamu
    resiko_usaha_TINGGI = 1 if resiko_usaha == "Tinggi" else 0
    resiko_usaha_SEDANG = 1 if resiko_usaha == "Sedang" else 0

st.markdown("---")

# 4. Tombol untuk Eksekusi
if st.button("Analisis Kelayakan"):
    # Susun jawaban inputan dari web
    data_input = {
        'plafond_fix': plafond_fix,
        'tenor_fix': tenor_fix,
        'resiko_usaha_TINGGI': resiko_usaha_TINGGI,
        'resiko_usaha_SEDANG': resiko_usaha_SEDANG
        # PENTING: Pastikan semua kolom yang ada di 'list_fitur' dimasukkan di sini
    }
    
    # Jadikan DataFrame
    data_baru = pd.DataFrame([data_input])
    
    # Pastikan urutan kolom sesuai dengan saat training
    # Ini gunanya kita menyimpan 'list_fitur' di dalam paket
    data_baru = data_baru[list_fitur] 
    
    # 5. PENTING: Lakukan standarisasi/scaling sebelum diprediksi!
    data_scaled = scaler.transform(data_baru)
    
    # 6. Lakukan prediksi menggunakan model CART
    prediksi = model.predict(data_scaled)
    
    # Tampilkan Hasil
    if prediksi[0] == 'ACC' or prediksi[0] == 1:
        st.success("🟢 HASIL: Calon Debitur Layak (Zona Hijau / ACC)")
    else:
        st.error("🔴 HASIL: Calon Debitur Berisiko Tinggi (Zona Merah / TOLAK)")
