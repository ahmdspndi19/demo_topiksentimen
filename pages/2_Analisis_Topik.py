# pages/2_Analisis_Topik.py
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import get_data # Standar impor

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Topik",
    page_icon="🏷️",
    layout="wide"
)

# --- JUDUL DAN DESKRIPSI ---
st.title("🏷️ Analisis Topik Mendalam")
st.markdown("Jelajahi topik-topik yang paling sering dibicarakan oleh pengguna dan sentimen yang terkait.")
st.markdown("---")

# --- MEMUAT DATA ---
df = get_data()

# --- VALIDASI DATA ---
if df is None or df.empty:
    st.warning("Data tidak tersedia. Tidak dapat menampilkan analisis topik.")
    st.stop()

# --- ANALISIS SENTIMEN PER TOPIK ---
st.header("Distribusi Sentimen per Topik")
# Menggunakan kolom 'Deskripsi Topik'
all_topics = df['Deskripsi Topik'].unique().tolist()
selected_topic = st.selectbox("Pilih Topik Utama untuk dianalisis:", options=all_topics)

if selected_topic:
    # Melakukan filter berdasarkan 'Deskripsi Topik'
    df_topic = df[df['Deskripsi Topik'] == selected_topic]
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Word Cloud untuk Topik: '{selected_topic}'")
        # Kolom 'Topik_Gabungan' sudah dibuat di utils.py, jadi ini tidak perlu diubah
        text = " ".join(ulasan for ulasan in df_topic['Topik_Gabungan'].astype(str).dropna())
        
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', collocations=False).generate(text)
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)
        else:
            st.info("Tidak ada kata kunci yang cukup untuk membuat Word Cloud pada topik ini.")

    with col2:
        st.subheader("Distribusi Sentimen")
        # Menggunakan kolom 'Sentimen'
        sentiment_counts_topic = df_topic['Sentimen'].value_counts(normalize=True).mul(100).rename('persentase').reset_index()
        fig_pie_topic = px.pie(sentiment_counts_topic,
                               names='Sentimen', # Menggunakan 'Sentimen'
                               values='persentase',
                               title=f'Sentimen di Topik "{selected_topic}"',
                               color='Sentimen', # Menggunakan 'Sentimen'
                               color_discrete_map={'Positif':'#28a745', 'Negatif':'#dc3545', 'Netral':'#ffc107'})
        fig_pie_topic.update_traces(textposition='inside', textinfo='percent+label')
        
        # Buat dictionary config untuk opsi Plotly
        plotly_config = {
            'displayModeBar': False, # Contoh: menyembunyikan menu bar plotly
            'scrollZoom': True
        }
        
        st.plotly_chart(fig_pie_topic, width='stretch', config=plotly_config)

    st.subheader("Contoh Ulasan Terkait")
    # Menampilkan kolom 'Kalimat', 'Sentimen', dan 'Detail Topik'
    st.dataframe(df_topic[['Kalimat', 'Sentimen', 'Detail Topik']].head(10))