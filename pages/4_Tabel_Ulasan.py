# pages/4_Tabel_Ulasan.py
import streamlit as st
import pandas as pd
from utils import get_data # Standar impor

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Explorer Ulasan",
    page_icon="📝",
    layout="wide"
)

# --- JUDUL DAN DESKRIPSI ---
st.title("📝 Explorer Data Ulasan")
st.markdown("Cari, filter, dan lihat detail setiap ulasan yang telah dianalisis.")
st.markdown("---")

# --- MEMUAT DATA ---
df = get_data()

# --- VALIDASI DATA ---
if df is None or df.empty:
    st.warning("Data tidak tersedia. Tidak dapat menampilkan tabel ulasan.")
    st.stop()

# --- OPSI FILTER DI SIDEBAR ---
st.sidebar.header("Opsi Filter")

# Filter berdasarkan Sentimen
# Menggunakan kolom 'Sentimen'
sentiments_options = ['Semua'] + df['Sentimen'].unique().tolist()
selected_sentiment = st.sidebar.selectbox("Filter berdasarkan Sentimen:", options=sentiments_options)

# Filter berdasarkan Topik
# Menggunakan kolom 'Deskripsi Topik'
topics_options = ['Semua'] + df['Deskripsi Topik'].unique().tolist()
selected_topic = st.sidebar.selectbox("Filter berdasarkan Topik:", options=topics_options)

# Pencarian teks
# Menggunakan kolom 'Kalimat' untuk pencarian
search_query = st.sidebar.text_input("Cari kata kunci dalam ulasan:")

# --- LOGIKA FILTER ---
df_selection = df.copy() # Mulai dengan semua data

if selected_sentiment != 'Semua':
    df_selection = df_selection[df_selection['Sentimen'] == selected_sentiment]

if selected_topic != 'Semua':
    df_selection = df_selection[df_selection['Deskripsi Topik'] == selected_topic]

if search_query:
    df_selection = df_selection[df_selection['Kalimat'].str.contains(search_query, case=False, na=False)]

# --- TAMPILKAN HASIL ---
st.info(f"Menampilkan {len(df_selection)} dari {len(df)} total ulasan berdasarkan filter Anda.")

# Tampilkan tabel data yang sudah difilter
# Menyesuaikan daftar kolom yang akan ditampilkan
columns_to_display = ['Kalimat', 'Sentimen', 'Deskripsi Topik', 'Detail Topik']
if 'confidence_score' in df_selection.columns:
    columns_to_display.append('confidence_score')

st.dataframe(
    df_selection[columns_to_display],
    use_container_width=True,
    hide_index=True,
    column_config={
        "confidence_score": st.column_config.ProgressColumn(
            "Skor Kepercayaan",
            format="%.2f",
            min_value=0,
            max_value=1,
        ),
    }
)