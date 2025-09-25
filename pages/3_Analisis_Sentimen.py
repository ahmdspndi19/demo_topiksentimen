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

# Definisikan dictionary config sekali di awal untuk digunakan kembali
plotly_config = {
    'displayModeBar': False,
    'scrollZoom': True
}

# --- VISUALISASI UTAMA ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Sentimen Keseluruhan")
    # Menggunakan kolom 'Sentimen'
    sentiment_counts = df['Sentimen'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentimen', 'jumlah']

    # Menggunakan plotly.express.bar untuk pemetaan warna
    fig_bar = px.bar(
        sentiment_counts,
        x='Sentimen', # Menggunakan 'Sentimen'
        y='jumlah',
        color='Sentimen', # Mewarnai bar berdasarkan kolom 'Sentimen'
        title='Jumlah Ulasan per Kategori Sentimen',
        text='jumlah',
        color_discrete_map={
            'Positif': '#28a745',
            'Negatif': '#dc3545',
            'Netral': '#ffc107'
        }
    )
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar, width='stretch', config=plotly_config)

with col2:
    st.subheader("Distribusi Sentimen Keseluruhan")
    # Menggunakan kolom 'Sentimen' untuk menghitung jumlah
    sentiment_counts = df['Sentimen'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentimen', 'jumlah']

    # Menggunakan plotly.express.pie untuk membuat diagram lingkaran
    fig_pie = px.pie(
        sentiment_counts,
        names='Sentimen',      # Kolom untuk label setiap irisan
        values='jumlah',       # Kolom untuk nilai/ukuran setiap irisan
        color='Sentimen',      # Mewarnai irisan berdasarkan kolom 'Sentimen'
        title='Proporsi Ulasan per Kategori Sentimen',
        color_discrete_map={
            'Positif': '#28a745',
            'Negatif': '#dc3545',
            'Netral': '#ffc107'
        }
    )
    # Menambahkan persentase dan label pada setiap irisan
    fig_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie, width='stretch', config=plotly_config)

# --- ANALISIS TOPIK PER SENTIMEN ---
st.header("Analisis Topik per Kategori Sentimen")
selected_sentiment = st.selectbox(
    "Pilih kategori sentimen untuk melihat topik terkait:",
    options=['Positif', 'Netral', 'Negatif']
)

if selected_sentiment:
    st.subheader(f"Topik yang Paling Sering Muncul untuk Sentimen '{selected_sentiment}'")
    # Melakukan filter berdasarkan kolom 'Sentimen'
    df_sentiment = df[df['Sentimen'] == selected_sentiment]
    
    if not df_sentiment.empty:
        # Mengambil topik dari kolom 'Deskripsi Topik'
        top_topics_sentiment = df_sentiment['Deskripsi Topik'].value_counts().nlargest(10).reset_index()
        top_topics_sentiment.columns = ['Topik', 'Jumlah']
        
        fig_bar_sentiment = px.bar(top_topics_sentiment,
                                       x='Jumlah',
                                       y='Topik',
                                       orientation='h',
                                       title=f'Top 10 Topik Sentimen {selected_sentiment}',
                                       text='Jumlah')
        fig_bar_sentiment.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar_sentiment, width='stretch', config=plotly_config)
    else:
        st.info(f"Tidak ada ulasan dengan sentimen '{selected_sentiment}' untuk dianalisis.")