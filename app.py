import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- 1. KONFIGURASI LAYAR LEBAR ---
st.set_page_config(page_title="Dashboard Analisis Risiko Kredit", layout="wide")

# --- 2. LOAD MODEL PREDIKSI (.PKL) ---
paket_otak = joblib.load('credit_risk_model_CART.pkl')
model = paket_otak['model']
scaler = paket_otak['scaler']
list_fitur = paket_otak['list_fitur']

# --- 3. LOAD DATASET HISTORIS ---
try:
    # Menggunakan file final yang sudah digabung
    df = pd.read_excel('Laporan_Final_CreditRisk (5).xlsx')
except FileNotFoundError:
    st.error("File Laporan_Final_CreditRisk (5) belum terdeteksi di GitHub. Pastikan nama dan ekstensinya pas!")
    df = pd.DataFrame()

# ==========================================
# SIDEBAR (PANEL KIRI UNTUK INPUT PREDIKSI)
# ==========================================
st.sidebar.header("Form Prediksi CART")

plafond_fix = st.sidebar.number_input("Input plafond_fix", value=15000000.0)
tenor_fix = st.sidebar.number_input("Input tenor_fix", value=3.0)
resiko_usaha = st.sidebar.selectbox("Input Risiko Usaha", ["SANGAT RENDAH", "RENDAH", "SEDANG", "TINGGI", "SANGAT TINGGI"])

resiko_usaha_TINGGI = 1 if resiko_usaha in ["TINGGI", "SANGAT TINGGI"] else 0
resiko_usaha_SEDANG = 1 if resiko_usaha == "SEDANG" else 0

hasil_tebakan = "-"
warna_tebakan = "black"

if st.sidebar.button("Analisis Kelayakan"):
    data_input = {fitur: 0 for fitur in list_fitur}
    
    if 'plafond_fix' in list_fitur: data_input['plafond_fix'] = plafond_fix
    if 'tenor_fix' in list_fitur: data_input['tenor_fix'] = tenor_fix
    if 'resiko_usaha_TINGGI' in list_fitur: data_input['resiko_usaha_TINGGI'] = resiko_usaha_TINGGI
    if 'resiko_usaha_SEDANG' in list_fitur: data_input['resiko_usaha_SEDANG'] = resiko_usaha_SEDANG
    
    data_baru = pd.DataFrame([data_input])[list_fitur]
    data_scaled = scaler.transform(data_baru)
    prediksi = model.predict(data_scaled)
    
    if prediksi[0] == 'ACC' or prediksi[0] == 1:
        hasil_tebakan = "Lancar (ACC)"
        warna_tebakan = "green"
    else:
        hasil_tebakan = "Berisiko (TOLAK)"
        warna_tebakan = "red"

st.sidebar.markdown(f"### Hasil: <span style='color:{warna_tebakan}'>{hasil_tebakan}</span>", unsafe_allow_html=True)

# ==========================================
# BAGIAN UTAMA (HEADER & KARTU ANGKA KPI)
# ==========================================
st.title("Dashboard Analisis Risiko Kredit & Klasifikasi Calon Debitur")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_project = len(df) if not df.empty else 0

kpi1.metric(label="Rata Rata Pinjaman", value="Rp167M") 
kpi2.metric(label="Total Project", value=f"{total_project:,.0f}")
kpi3.metric(label="Keputusan System Kredit (ACC)", value="4,760")
kpi4.metric(label="Hasil Prediksi CART (Live)", value=hasil_tebakan)

st.markdown("---")

# ==========================================
# BAGIAN GRAFIK (DI BAWAH KPI)
# ==========================================
col_kiri, col_kanan = st.columns(2)

with col_kiri:
    st.subheader("Distribusi Risiko Usaha")
    if not df.empty:
        try:
            fig_pie = px.pie(df, names='resiko_usaha', hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Error Grafik Kiri: {e}")

with col_kanan:
    st.subheader("Distribusi Keputusan Berdasarkan Profil Risiko")
    if not df.empty:
        try:
            # SUDAH DIREVISI: Menggunakan 'status' sesuai gambar Excel kamu
            df_bar = df.groupby(['resiko_usaha', 'status']).size().reset_index(name='Jumlah')
            
            fig_bar = px.bar(df_bar, y='resiko_usaha', x='Jumlah', color='status', 
                             orientation='h', barmode='group',
                             color_discrete_map={'ACC': '#28a745', 'TOLAK': '#dc3545'}) 
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Error Grafik Kanan: {e}")
