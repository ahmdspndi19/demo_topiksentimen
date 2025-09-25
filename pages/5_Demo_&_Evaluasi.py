# pages/5_Demo_&_Evaluasi.py

import streamlit as st
import time
import os
import re
import pandas as pd
import numpy as np
import pickle
import gensim
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from utils import load_text_file

# ==============================================================================
# FUNGSI-FUNGSI UNTUK MEMUAT ASET (DENGAN CACHING)
# ==============================================================================

@st.cache_resource
def muat_aset_pra_pemrosesan():
    """
    Memuat semua komponen yang dibutuhkan untuk pra-pemrosesan teks.
    Fungsi ini menggunakan cache agar tidak dimuat berulang kali.
    """
    try:
        # Sastrawi
        stemmer_factory = StemmerFactory()
        stemmer = stemmer_factory.create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        stopword_remover = stopword_factory.create_stop_word_remover()

        # Kamus Normalisasi
        kamus_df = pd.read_excel("assets/kamuskatabaku.xlsx")
        kamus_norm = dict(zip(kamus_df['tidak_baku'], kamus_df['kata_baku']))
        
        # Custom Stopwords untuk LDA
        custom_stopwords = [
                    'aman', 'aneh', 'bagus', 'baik', 'bingung', 'buruk', 'cepat', 'cocok', 'enak',
            'efisien', 'gagal', 'gercep', 'hebat', 'hemat', 'jelek', 'jelas', 'kecewa',
            'keren', 'lancar', 'lambat', 'lemot', 'lumayan', 'mahal', 'malas', 'mantap',
            'mantul', 'membantu', 'mending', 'mudah', 'nyaman', 'ok', 'paham', 'parah',
            'praktis', 'puas', 'pusing', 'repot', 'ribet', 'rumit', 'salah', 'sesuai', 'sip',
            'sukses', 'sulit', 'super', 'susah', 'takut', 'terbantu', 'top', 'simpel',
            'amanah', 'bintang', 'mantab', 'trimakasih', 'gampang', 'oke', 'good', 'muas',
            'percaya', 'eror', 'error', 'pungli', 'kesah', 'mulu', 'sayang', 'simple',
            'manfaat', 'best', 'the', 'efektif', 'adu', 'capek', 'ganggu', 'calo', 'terang',
            'manis', 'job', 'bebas', 'saran', 'taat', 'admin', 'anda', 'bro', 'gan', 'kak',
            'kakak', 'kamu', 'min', 'saya', 'sis', 'aja', 'banget', 'bolak', 'coba', 'deh',
            'dgn', 'dong', 'ga', 'gak', 'gk', 'enggak', 'kalo', 'kalau', 'kah', 'kayak',
            'kok', 'krn', 'line', 'nggak', 'nih', 'on', 'sih', 'yg', 'ya', 'si', 'biar',
            'cuman', 'n', 'satset', 'doang', 'app', 'aplikasi', 'apk', 'jadi', 'banyak',
            'hasil', 'metode', 'kerja', 'pakai', 'alhamdulillah', 'terima_kasih', 'thanks',
            'good_job', 'the_best', 'sat_set', 'terima', 'kasih', 'mohon', 'moga', 'semoga',
            'tolong', 'tinggal', 'tunggu'
        ]

        # Kamus Label Topik Deskriptif (disesuaikan agar cocok dengan output laporan)
        label_topik = {
            0: "Fitur dan Layanan Bantuan Aplikasi",
            1: "Proses dan Durasi Pengiriman STNK",
            2: "Kualitas Pelayanan dan Antrean di Samsat",
            3: "Layanan Pengiriman STNK ke Rumah",
            4: "Pembayaran Pajak Kendaraan Online",
            5: "Permintaan Bantuan dan Saran Peningkatan",
            6: "Rincian Biaya Administrasi dan Transaksi",
            7: "Kendala Teknis Aplikasi (Loading)",
            8: "Proses Pengurusan via Aplikasi SIGNAL",
            9: "Layanan Jasa Kurir Pengiriman",
            10: "Perpanjangan STNK 5 Tahunan (Ganti Plat)",
            11: "Perbandingan Layanan Online vs. Kantor Samsat"
        }

        return stemmer, stopword_remover, kamus_norm, custom_stopwords, label_topik
    except Exception as e:
        st.error(f"Gagal memuat aset pra-pemrosesan: {e}")
        return None, None, None, None, None

@st.cache_resource
def muat_model_sentimen():
    """Memuat model sentimen, tokenizer, dan label encoder."""
    try:
        model = load_model("assets/models/model_sentimen_lstm.h5")
        with open("assets/models/tokenizer.pkl", 'rb') as f: tokenizer = pickle.load(f)
        with open("assets/models/label_encoder.pkl", 'rb') as f: label_encoder = pickle.load(f)
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Gagal memuat model sentimen dari 'assets/': {e}")
        return None, None, None

@st.cache_resource
def muat_model_lda():
    """Memuat model LDA dan dictionary 12 topik."""
    try:
        model = gensim.models.LdaModel.load("assets/models/model_lda_terbaik_12topik.model")
        dictionary = gensim.corpora.Dictionary.load("assets/models/model_lda_terbaik_12topik.dict")
        return model, dictionary
    except Exception as e:
        st.error(f"Gagal memuat model LDA 12 topik dari 'assets/models': {e}")
        return None, None

# ==============================================================================
# FUNGSI-FUNGSI PEMROSESAN DAN PREDIKSI
# ==============================================================================

def pra_pemrosesan_lstm(teks, kamus_norm, stemmer_obj, stopword_remover_obj):
    teks = teks.lower()
    words = teks.split()
    normalized_words = [kamus_norm.get(word, word) for word in words]
    teks = ' '.join(normalized_words)
    teks = stopword_remover_obj.remove(teks)
    teks = stemmer_obj.stem(teks)
    return teks

def pra_pemrosesan_lda(teks, kamus_norm, stemmer_obj, custom_stopwords_list):
    teks = teks.lower()
    words = teks.split()
    normalized_words = [kamus_norm.get(word, word) for word in words]
    teks = ' '.join(normalized_words)
    words = [word for word in teks.split() if word not in custom_stopwords_list]
    teks = ' '.join(words)
    teks = stemmer_obj.stem(teks)
    return teks

def prediksi_sentimen(teks, model, tokenizer, label_encoder, aset_prep):
    stemmer, stopword, kamus, _, _ = aset_prep
    teks_bersih = pra_pemrosesan_lstm(teks, kamus, stemmer, stopword)
    if not teks_bersih: return "Netral", 0.5
    
    maxlen = model.input_shape[1]
    sekuens = tokenizer.texts_to_sequences([teks_bersih])
    padded = pad_sequences(sekuens, maxlen=maxlen, padding='post', truncating='post')
    
    prediksi_prob = model.predict(padded, verbose=0)
    skor_keyakinan = np.max(prediksi_prob)
    prediksi_kelas = np.argmax(prediksi_prob, axis=1)
    label_prediksi = label_encoder.inverse_transform(prediksi_kelas)[0]
    
    return label_prediksi, skor_keyakinan

def prediksi_topik_lda(teks, model, dictionary, aset_prep):
    stemmer, _, kamus, custom_stopwords, label_topik_map = aset_prep
    teks_bersih = pra_pemrosesan_lda(teks, kamus, stemmer, custom_stopwords)
    if not teks_bersih: return "Topik Tidak Relevan", "Teks kosong...", 0.0
    tokens = teks_bersih.split()
    bow_vector = dictionary.doc2bow(tokens)
    topic_distribution = model.get_document_topics(bow_vector, minimum_probability=0.1)
    if not topic_distribution: return "Topik Tidak Relevan", "Tidak ada topik cocok.", 0.0
    top_topic_index, confidence = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[0]
    deskripsi_topik = label_topik_map.get(top_topic_index, f"Topik {top_topic_index+1}")
    keywords_tuples = model.show_topic(top_topic_index, topn=5)
    keywords_string = ', '.join([word for word, prop in keywords_tuples])
    return deskripsi_topik, keywords_string, confidence

def parse_lda_report(report_text, labels_map):
    """Mem-parsing teks laporan LDA untuk mengekstrak informasi kunci."""
    if not report_text or "File tidak ditemukan" in report_text: return {}
    results = {}
    try:
        # Mengambil metrik utama
        coherence_match = re.search(r"Skor koherensi tertinggi adalah ([\d.]+)", report_text)
        if coherence_match: results['coherence_score'] = float(coherence_match.group(1))
        
        topics_match = re.search(r"Jumlah topik yang menghasilkan skor tertinggi: (\d+)", report_text)
        if topics_match: results['optimal_topics'] = int(topics_match.group(1))
        
        # Ekstrak topik dan kata kuncinya
        topics_section = re.search(r"Topik yang ditemukan oleh model dengan \d+ topik:([\s\S]*)====== EVALUASI FINAL ======", report_text)
        if topics_section:
            topics_list = topics_section.group(1).strip().split('\n')
            topics_dict = {}
            for i, line in enumerate(topics_list):
                if 'Topik' in line:
                    # Mengambil hanya kata-kata kunci
                    keywords_raw = line.split(': ')[1]
                    keywords = [re.sub(r'[\d."*+]', '', word).strip() for word in keywords_raw.split('+')]
                    # Menggunakan label deskriptif dari map
                    topic_label = labels_map.get(i, f"Topik {i+1}")
                    topics_dict[topic_label] = keywords
            results['topics'] = topics_dict
    except Exception as e:
        st.error(f"Gagal mem-parsing laporan LDA: {e}")
    return results

# ==============================================================================
# KONFIGURASI DAN TATA LETAK HALAMAN STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Demo & Evaluasi Model", page_icon="🚀", layout="wide")

# --- Muat Semua Aset di Awal ---
aset_prep = muat_aset_pra_pemrosesan()
model_sentimen, tokenizer_sentimen, le_sentimen = muat_model_sentimen()
model_lda, dictionary_lda = muat_model_lda()

# --- Judul Halaman ---
st.title("🚀 Demo Interaktif & Evaluasi Model")
st.markdown("Halaman ini berisi demo interaktif untuk analisis sentimen dan pemodelan topik, serta hasil evaluasi model.")
st.markdown("---")

# --- BAGIAN DEMO INTERAKTIF (TERINTEGRASI) ---
st.header("🧪🔬 Demo Analisis Terintegrasi")

semua_aset_siap = all([model_sentimen, model_lda, aset_prep[0]])

if semua_aset_siap:
    input_teks = st.text_area(
        "Masukkan ulasan untuk dianalisis sentimen dan topiknya secara bersamaan:",
        "Aplikasinya bagus dan sangat membantu sekali untuk bayar pajak tahunan, jadi tidak usah antri lagi di samsat.",
        height=150,
        key="input_terintegrasi"
    )
    
    if st.button("Analisis Sekarang!", type="primary", key="btn_terintegrasi"):
        if input_teks.strip():
            with st.spinner("Menganalisis sentimen dan topik..."):
                hasil_sentimen, skor_sentimen = prediksi_sentimen(input_teks, model_sentimen, tokenizer_sentimen, le_sentimen, aset_prep)
                deskripsi_topik, keywords_topik, _ = prediksi_topik_lda(input_teks, model_lda, dictionary_lda, aset_prep)

            st.subheader("Hasil Analisis Gabungan")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Analisis Sentimen (LSTM)")
                if 'positif' in hasil_sentimen.lower():
                    st.success(f"**{hasil_sentimen.upper()}** 👍")
                elif 'negatif' in hasil_sentimen.lower():
                    st.error(f"**{hasil_sentimen.upper()}** 👎")
                else:
                    st.warning(f"**{hasil_sentimen.upper()}** 😐")
                st.caption(f"Tingkat Keyakinan: **{skor_sentimen:.2%}**")
                st.progress(float(skor_sentimen))
            with col2:
                st.markdown("##### Analisis Topik (LDA)")
                st.success(f"**{deskripsi_topik}**")
                st.caption(f"Kata Kunci: {keywords_topik}")
        else:
            st.warning("Mohon masukkan kalimat untuk dianalisis.")
else:
    st.error("Satu atau lebih model/aset gagal dimuat. Fitur demo interaktif tidak tersedia.")

st.markdown("---")

# --- BAGIAN EVALUASI STATIS ---
st.header("Evaluasi Kinerja Model (Statis)")

# Muat File Laporan
REPORTS_DIR = "assets/reports"
report_sebelum = load_text_file(os.path.join(REPORTS_DIR, "report_sebelum.txt"))
report_sesudah = load_text_file(os.path.join(REPORTS_DIR, "report_sesudah.txt"))
log_sebelum = load_text_file(os.path.join(REPORTS_DIR, "training_log_sebelum.txt"))
log_sesudah = load_text_file(os.path.join(REPORTS_DIR, "training_log_sesudah.txt"))
lda_report_full = load_text_file(os.path.join(REPORTS_DIR, "lda_report.txt"))

col1_eval, col2_eval = st.columns(2)
with col1_eval:
    st.subheader("📉 Model Sebelum: LSTM Sederhana")
    image_path_before = "assets/image_6737bc.png"
    if os.path.exists(image_path_before):
        st.image(image_path_before, caption="Confusion Matrix Model Dasar", width='stretch')
    else:
        st.warning(f"File gambar tidak ditemukan: '{image_path_before}'.")
    st.metric("Akurasi", "67,7%, ")
    with st.expander("Lihat Detail Analisis (Model Sebelum)"):
        st.code(report_sebelum, language='text')
        st.code(log_sebelum, language='text')
with col2_eval:
    st.subheader("📈 Model Sesudah: Stacked Bi-LSTM")
    image_path_after = "assets/image_673b45.png"
    if os.path.exists(image_path_after):
        st.image(image_path_after, caption="Confusion Matrix Model Optimal", width='stretch')
    else:
        st.warning(f"File gambar tidak ditemukan: '{image_path_after}'.")
    st.metric("Akurasi", "83,9%, ")
    with st.expander("Lihat Detail Analisis (Model Sesudah)"):
        st.code(report_sesudah, language='text')
        st.code(log_sesudah, language='text')

st.markdown("---")

# Tampilkan Analisis Pemodelan Topik (LDA)
st.header("Analisis Pemodelan Topik (LDA)")
_, _, _, _, label_topik = aset_prep
lda_results = parse_lda_report(lda_report_full, label_topik)

if not lda_results:
    st.error("File laporan `reports/lda_report.txt` tidak ditemukan atau gagal diparsing.")
else:
    col1_lda, col2_lda = st.columns(2)
    col1_lda.metric("Jumlah Topik Optimal", lda_results.get('optimal_topics', 'N/A'))
    col2_lda.metric("Skor Koherensi (C_v)", f"{lda_results.get('coherence_score', 0):.4f}")
    
    st.markdown("##### Visualisasi Penentuan Jumlah Topik Optimal")
    st.write(
        "Grafik di bawah ini menunjukkan skor koherensi dari proses validasi model. "
        "Puncak tertinggi pada **12 topik** dipilih sebagai konfigurasi model yang paling optimal."
    )
    
    lda_plot_path = "assets/lda_coherence_plot.png" 
    if os.path.exists(lda_plot_path):
        st.image(lda_plot_path, caption="Grafik Skor Koherensi (C_v) vs. Jumlah Topik", width='stretch')
    else:
        st.warning(f"File gambar '{lda_plot_path}' tidak ditemukan. Pastikan ada di folder 'assets'.")

    # Membuat tab untuk detail topik dan log proses
    tab_detail, tab_log = st.tabs(["Detail Topik & Kata Kunci", "Log Proses"])
    
    with tab_detail:
        st.subheader("Rincian Topik yang Ditemukan")
        if 'topics' in lda_results:
            for topic_name, keywords in lda_results['topics'].items():
                st.markdown(f"**{topic_name}**")
                st.text(", ".join(keywords))
                st.markdown("---")
        else:
            st.warning("Tidak dapat menampilkan detail topik dari laporan.")
            
    with tab_log:
        st.subheader("Log Lengkap Proses Pemodelan LDA")
        st.code(lda_report_full, language="text")