# app.py
import streamlit as st
from utils import get_data # Menggunakan fungsi terpusat

# --- KONFIGURASI HALAMAN ---
# Dijalankan pertama kali untuk mengatur tab browser dan layout
st.set_page_config(
    page_title="Selamat Datang | SentimenSignal",
    page_icon="👋",
    layout="wide",
)

# --- MEMUAT DATA ---
# Memanggil get_data() akan memastikan data dimuat ke session_state
# saat aplikasi pertama kali dijalankan.
df = get_data()

# --- KONTEN HALAMAN ---
st.title("Selamat Datang di SentimenSignal Dashboard 👋")
st.markdown("Platform untuk menganalisis sentimen dan mengekstrak topik utama dari ulasan pengguna secara otomatis.")
st.markdown("---")

# --- BAGIAN PENJELASAN DASHBOARD ---
st.header("Tentang Dashboard Ini")
st.markdown("""
Dashboard ini dirancang untuk memberikan wawasan mendalam dari data ulasan produk. Dengan memanfaatkan model *Machine Learning*, kami mengklasifikasikan sentimen (Positif, Negatif, Netral) dan mengidentifikasi topik yang paling sering dibicarakan oleh pengguna.

Tujuannya adalah untuk membantu tim produk, pemasaran, dan layanan pelanggan dalam memahami suara konsumen secara efisien dan akurat.
""")

# --- BAGIAN PENJELASAN DATA ---
st.header("Sumber Data")
if df is not None and not df.empty:
    total_reviews_info = len(df)
    st.info(f"Analisis ini didasarkan pada **{total_reviews_info:,} ulasan** yang telah diproses.")
else:
    # Pesan ini hanya akan muncul jika get_data() gagal
    st.error("Data ulasan tidak dapat dimuat. Pastikan file `hasil_terstruktur_diperbaiki.csv` ada di direktori utama.")

# --- BAGIAN CARA PENGGUNAAN ---
st.header("Bagaimana Cara Menggunakan Dashboard Ini?")
st.markdown("""
Navigasikan melalui halaman-halaman yang tersedia di **sidebar kiri** untuk menjelajahi berbagai aspek analisis:

1.  **Ringkasan Analisis**: Memberikan ringkasan statistik dan *key insights* dari keseluruhan data ulasan.
2.  **Analisis Topik**: Menyelami topik-topik spesifik, lengkap dengan *word cloud* dan distribusi sentimen per topik.
3.  **Analisis Sentimen**: Visualisasi distribusi sentimen secara keseluruhan dan berdasarkan skor kepercayaan model.
4.  **Tabel Ulasan**: Memungkinkan Anda untuk mencari, memfilter, dan melihat data ulasan mentah secara detail.
5.  **Demo & Evaluasi**: Mencoba demo interaktif model dan melihat metrik evaluasi kinerjanya.
""")

# Menambahkan gambar ilustratif
st.image("https://storage.googleapis.com/gweb-uniblog-publish-prod/images/New-blog-header_61.width-1200.format-webp.webp",
         caption="Visualisasi Analisis Data untuk Pengambilan Keputusan")