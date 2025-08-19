# pages/1_Halaman_Utama.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_data # Standar impor

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Ringkasan Analisis",
    page_icon="📊",
    layout="wide"
)

# --- JUDUL DAN DESKRIPSI ---
st.title("📊 Ringkasan Analisis Keseluruhan")
st.markdown("Halaman ini menyajikan gambaran umum dari data ulasan yang dianalisis.")
st.markdown("---")

# --- MEMUAT DATA ---
df = get_data()

# --- VALIDASI DATA ---
if df is None or df.empty:
    st.warning("Data tidak tersedia. Tidak dapat menampilkan ringkasan. Silakan periksa file data Anda.")
    st.stop() # Menghentikan eksekusi jika tidak ada data

# --- METRIK UTAMA ---
st.header("Metrik Utama")
col1, col2, col3, col4 = st.columns(4)

total_reviews = len(df)
positif_count = int(df[df['sentimen'] == 'Positif'].shape[0])
netral_count = int(df[df['sentimen'] == 'Netral'].shape[0])
negatif_count = int(df[df['sentimen'] == 'Negatif'].shape[0])

col1.metric("Total Ulasan", f"{total_reviews:,}")
col2.metric("Sentimen Positif", f"{positif_count:,}", f"{positif_count/total_reviews:.1%}")
col3.metric("Sentimen Netral", f"{netral_count:,}", f"{netral_count/total_reviews:.1%}")
col4.metric("Sentimen Negatif", f"{negatif_count:,}", f"-{negatif_count/total_reviews:.1%}")

st.markdown("---")

# --- VISUALISASI ---
st.header("Visualisasi Data")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Sentimen")
    # --- PERBAIKAN DI SINI ---
    # Biarkan pandas menamai kolomnya secara otomatis ('sentimen' dan 'count')
    sentiment_counts = df['sentimen'].value_counts().reset_index()
    
    # Gunakan nama kolom 'count' yang benar untuk parameter 'values'
    fig_pie = px.pie(sentiment_counts,
                     names='sentimen',
                     values='count', # <-- Menggunakan nama kolom yang benar
                     title='Proporsi Sentimen Ulasan',
                     color='sentimen',
                     color_discrete_map={'Positif':'#28a745', 'Negatif':'#dc3545', 'Netral':'#ffc107'})
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Topik Paling Umum")
    # Kode ini sudah benar, kita hanya standarisasi nama kolom untuk konsistensi
    top_topics = df['deskripsi_topik'].value_counts().nlargest(10).reset_index()
    fig_bar = px.bar(top_topics,
                     x='count', # <-- Menggunakan nama kolom yang benar
                     y='deskripsi_topik',
                     orientation='h',
                     title='Top 10 Topik yang Paling Sering Muncul',
                     text='count',
                     labels={'deskripsi_topik': 'Topik', 'count': 'Jumlah'})
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)