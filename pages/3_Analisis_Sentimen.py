# pages/3_Analisis_Sentimen.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_data # Standar impor

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="😊",
    layout="wide"
)

# --- JUDUL DAN DESKRIPSI ---
st.title("😊 Analisis Sentimen Detail")
st.markdown("Visualisasi distribusi sentimen dan analisis berdasarkan skor kepercayaan model.")
st.markdown("---")

# --- MEMUAT DATA ---
df = get_data()

# --- VALIDASI DATA ---
if df is None or df.empty:
    st.warning("Data tidak tersedia. Tidak dapat menampilkan analisis sentimen.")
    st.stop()

# --- VISUALISASI UTAMA ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Sentimen Keseluruhan")
    sentiment_counts = df['sentimen'].value_counts().reset_index()
    sentiment_counts.columns = ['sentimen', 'jumlah']

    # --- PERBAIKAN DI SINI ---
    # Mengganti st.bar_chart dengan plotly.express.bar
    # untuk memungkinkan pemetaan warna berdasarkan kategori sentimen.
    fig_bar = px.bar(
        sentiment_counts,
        x='sentimen',
        y='jumlah',
        color='sentimen', # <-- Kunci untuk mewarnai bar berdasarkan kolom 'sentimen'
        title='Jumlah Ulasan per Kategori Sentimen',
        text='jumlah',
        color_discrete_map={
            'Positif': '#28a745',
            'Negatif': '#dc3545',
            'Netral': '#ffc107'
        }
    )
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)


with col2:
    st.subheader("Distribusi Skor Kepercayaan")
    if 'confidence_score' in df.columns:
        fig_hist = px.histogram(df, x="confidence_score", color="sentimen",
                                title="Distribusi Skor Kepercayaan berdasarkan Sentimen",
                                marginal="box",
                                color_discrete_map={'Positif':'#28a745', 'Negatif':'#dc3545', 'Netral':'#ffc107'})
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Kolom 'confidence_score' atau 'skor_sentimen' tidak tersedia untuk analisis ini.")

st.markdown("---")

# --- ANALISIS TOPIK PER SENTIMEN ---
st.header("Analisis Topik per Kategori Sentimen")
selected_sentiment = st.selectbox(
    "Pilih kategori sentimen untuk melihat topik terkait:",
    options=['Positif', 'Netral', 'Negatif']
)

if selected_sentiment:
    st.subheader(f"Topik yang Paling Sering Muncul untuk Sentimen '{selected_sentiment}'")
    df_sentiment = df[df['sentimen'] == selected_sentiment]
    
    if not df_sentiment.empty:
        top_topics_sentiment = df_sentiment['deskripsi_topik'].value_counts().nlargest(10).reset_index()
        top_topics_sentiment.columns = ['Topik', 'Jumlah']
        
        fig_bar_sentiment = px.bar(top_topics_sentiment,
                                   x='Jumlah',
                                   y='Topik',
                                   orientation='h',
                                   title=f'Top 10 Topik Sentimen {selected_sentiment}',
                                   text='Jumlah')
        fig_bar_sentiment.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar_sentiment, use_container_width=True)
    else:
        st.info(f"Tidak ada ulasan dengan sentimen '{selected_sentiment}' untuk dianalisis.")