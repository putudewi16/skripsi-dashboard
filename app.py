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
    # --- PERBAIKAN KEYERROR ---
    # 1. Bikin 'keranjang' data otomatis berisi angka 0 untuk semua fitur model
    data_input = {fitur: 0 for fitur in list_fitur}
    
    # 2. Update nilainya HANYA untuk variabel yang ada di form input dashboard
    # Menggunakan 'if' agar sistem tidak error jika ada sedikit perbedaan ejaan nama kolom
    if 'plafond_fix' in list_fitur: data_input['plafond_fix'] = plafond_fix
    if 'tenor_fix' in list_fitur: data_input['tenor_fix'] = tenor_fix
    if 'resiko_usaha_TINGGI' in list_fitur: data_input['resiko_usaha_TINGGI'] = resiko_usaha_TINGGI
    if 'resiko_usaha_SEDANG' in list_fitur: data_input['resiko_usaha_SEDANG'] = resiko_usaha_SEDANG
    
    # 3. Jadikan DataFrame
    data_baru = pd.DataFrame([data_input])
    
    # 4. Susun urutan kolom biar pas 100% dengan model (Error tidak akan muncul lagi di sini)
    data_baru = data_baru[list_fitur] 
    
    # 5. PENTING: Lakukan standarisasi/scaling sebelum diprediksi!
    data_scaled = scaler.transform(data_baru)
    
    # 6. Lakukan prediksi menggunakan model CART
    prediksi = model.predict(data_scaled)
    
    # 7. Tampilkan Hasil
    if prediksi[0] == 'ACC' or prediksi[0] == 1:
        st.success("🟢 HASIL: Calon Debitur Layak (Zona Hijau / ACC)")
    else:
        st.error("🔴 HASIL: Calon Debitur Berisiko Tinggi (Zona Merah / TOLAK)")
    
    # 5. PENTING: Lakukan standarisasi/scaling sebelum diprediksi!
    data_scaled = scaler.transform(data_baru)
    
    # 6. Lakukan prediksi menggunakan model CART
    prediksi = model.predict(data_scaled)
    
    # Tampilkan Hasil
    if prediksi[0] == 'ACC' or prediksi[0] == 1:
        st.success("🟢 HASIL: Calon Debitur Layak (Zona Hijau / ACC)")
    else:
        st.error("🔴 HASIL: Calon Debitur Berisiko Tinggi (Zona Merah / TOLAK)")
