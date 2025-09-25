# utils.py

import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data(filepath):
    """
    Memuat dan memproses data dari file CSV.
    Fungsi ini dijalankan sekali dan hasilnya di-cache.
    """
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        st.error(f"File data tidak ditemukan di path: {abs_path}")
        st.warning(f"Pastikan file '{os.path.basename(filepath)}' ada di direktori utama proyek.")
        st.info(f"Direktori kerja saat ini: {os.getcwd()}")
        return None

    try:
        df = pd.read_csv(filepath)
        # Ganti nama 'Skor Sentimen' menjadi 'confidence_score' untuk konsistensi internal
        # Menyesuaikan nama kolom sumber menjadi 'Skor Sentimen'
        if 'Skor Sentimen' in df.columns:
            df.rename(columns={'Skor Sentimen': 'confidence_score'}, inplace=True)

        # Menyesuaikan nama kolom 'deskripsi_topik' dan 'detail_topik'
        df['Topik_Gabungan'] = df['Deskripsi Topik'].astype(str) + ", " + df['Detail Topik'].astype(str)
        
        # Menyesuaikan nama kolom 'ulasan_lengkap' menjadi 'Kalimat'
        if 'pecahan_kalimat' not in df.columns:
            df['pecahan_kalimat'] = df['Kalimat']
        return df
    except KeyError as e:
        st.error(f"Terjadi kesalahan: Kolom yang diharapkan tidak ditemukan -> {e}.")
        st.info("Pastikan file CSV Anda memiliki kolom: 'Kalimat', 'Deskripsi Topik', 'Detail Topik', 'Sentimen', 'Skor Sentimen'.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
        return None

def get_data():
    """
    Fungsi andal untuk mendapatkan data.
    Memuat data menggunakan load_data() dan menyimpannya di st.session_state.
    Ini memastikan data hanya dimuat sekali per sesi.
    """
    if 'reviews_data' not in st.session_state:
        st.session_state.reviews_data = load_data('hasil_terstruktur_diperbaiki.csv')
    return st.session_state.reviews_data

def get_sentiment_badge(sentiment):
    """Mengembalikan warna badge untuk sentimen."""
    if sentiment == 'Positif':
        return 'green'
    elif sentiment == 'Netral':
        return 'orange'
    elif sentiment == 'Negatif':
        return 'red'
    return 'grey'

def load_text_file(filepath):
    """Membaca konten dari sebuah file teks."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"File tidak ditemukan: {filepath}"