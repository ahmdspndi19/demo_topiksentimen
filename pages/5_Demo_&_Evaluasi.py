# pages/5_Demo_&_Evaluasi.py
import streamlit as st
import time
import os
import re
import pandas as pd
from utils import load_text_file

# --- FUNGSI BANTU UNTUK PARSING LAPORAN LDA ---
def parse_lda_report(report_text):
    """
    Mem-parsing teks laporan LDA untuk mengekstrak informasi kunci.
    """
    if not report_text or "File tidak ditemukan" in report_text:
        return {}

    results = {}
    try:
        coherence_match = re.search(r"Skor Koherensi \(C_v\) Tertinggi: ([\d.]+)", report_text)
        if coherence_match:
            results['coherence_score'] = float(coherence_match.group(1))

        topics_match = re.search(r"Jumlah Topik Optimal: (\d+)", report_text)
        if topics_match:
            results['optimal_topics'] = int(topics_match.group(1))

        alpha_match = re.search(r"- Alpha: ([\d\w.]+)", report_text)
        if alpha_match:
            results['alpha'] = alpha_match.group(1)
        
        eta_match = re.search(r"- Eta: ([\d\w.]+)", report_text)
        if eta_match:
            results['eta'] = eta_match.group(1)

        topics_keywords_raw = re.search(r"Topik yang ditemukan oleh model terbaik \(hanya keywords\):([\s\S]*)Topik yang ditemukan \(dengan format asli", report_text)
        if topics_keywords_raw:
            topics_list = topics_keywords_raw.group(1).strip().split('\n')
            topics_dict = {}
            for line in topics_list:
                if ': ' in line:
                    parts = line.split(': ')
                    topic_name = parts[0].strip()
                    keywords = [kw.strip() for kw in parts[1].split(',')]
                    topics_dict[topic_name] = keywords
            results['topics'] = topics_dict
    except Exception as e:
        st.error(f"Gagal mem-parsing laporan LDA: {e}")
    
    return results

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Evaluasi & Analisis",
    page_icon="🚀",
    layout="wide"
)

# --- JUDUL DAN DESKRIPSI ---
st.title("🚀 Evaluasi Kinerja & Analisis Model")
st.markdown("Halaman ini berisi evaluasi untuk model klasifikasi sentimen dan hasil dari pemodelan topik (LDA).")
st.markdown("---")

# --- MEMUAT SEMUA LAPORAN DARI FILE ---
REPORTS_DIR = "assets/reports"
report_sebelum = load_text_file(os.path.join(REPORTS_DIR, "report_sebelum.txt"))
report_sesudah = load_text_file(os.path.join(REPORTS_DIR, "report_sesudah.txt"))
log_sebelum = load_text_file(os.path.join(REPORTS_DIR, "training_log_sebelum.txt"))
log_sesudah = load_text_file(os.path.join(REPORTS_DIR, "training_log_sesudah.txt"))
lda_report_full = load_text_file(os.path.join(REPORTS_DIR, "lda_report.txt"))

# ==============================================================================
# BAGIAN EVALUASI MODEL KLASIFIKASI SENTIMEN
# ==============================================================================
st.header("Evaluasi Model Klasifikasi Sentimen")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Model Sebelum: LSTM Sederhana")
    image_path_before = "assets/image_6737bc.png"
    if os.path.exists(image_path_before):
        st.image(image_path_before, caption="Confusion Matrix Model Dasar", use_container_width=True)
    else:
        st.warning(f"File gambar tidak ditemukan: '{image_path_before}'.")
    st.metric("Akurasi", "67.5%", delta="-16.6%", delta_color="inverse")
    with st.expander("Lihat Detail Analisis (Model Sebelum)"):
        st.code(report_sebelum, language='text')
        st.code(log_sebelum, language='text')

with col2:
    st.subheader("📈 Model Sesudah: Stacked Bi-LSTM")
    image_path_after = "assets/image_673b45.png"
    if os.path.exists(image_path_after):
        st.image(image_path_after, caption="Confusion Matrix Model Optimal", use_container_width=True)
    else:
        st.warning(f"File gambar tidak ditemukan: '{image_path_after}'.")
    st.metric("Akurasi", "84.1%", delta="16.6%")
    with st.expander("Lihat Detail Analisis (Model Sesudah)"):
        st.code(report_sesudah, language='text')
        st.code(log_sesudah, language='text')

st.markdown("---")

# ==============================================================================
# BAGIAN ANALISIS PEMODELAN TOPIK (LDA)
# ==============================================================================
st.header("Analisis Pemodelan Topik (LDA)")

lda_results = parse_lda_report(lda_report_full)

if not lda_results:
    st.error("File laporan `reports/lda_report.txt` tidak ditemukan atau gagal diparsing.")
    st.stop()

col1_lda, col2_lda, col3_lda = st.columns(3)
col1_lda.metric("Jumlah Topik Optimal", lda_results.get('optimal_topics', 'N/A'))
col2_lda.metric("Skor Koherensi (C_v)", f"{lda_results.get('coherence_score', 0):.4f}")
col3_lda.metric("Hyperparameter (α / η)", f"{lda_results.get('alpha', 'N/A')} / {lda_results.get('eta', 'N/A')}")

tab1, tab2, tab3 = st.tabs(["Visualisasi LDA (pyLDAvis)", "Detail Topik & Kata Kunci", "Log Proses"])

with tab1:
    st.subheader("Visualisasi Interaktif Model Topik")
    
    # --- PERBAIKAN DI SINI ---
    # Kode sekarang akan secara otomatis memuat dan menampilkan file HTML Anda
    vis_html_path = "assets/lda_visualization.html"
    if os.path.exists(vis_html_path):
        with open(vis_html_path, 'r', encoding='utf-8') as f:
            html_string = f.read()
        st.components.v1.html(html_string, width=1300, height=800, scrolling=True)
        st.success("Visualisasi LDA berhasil dimuat dari `assets/lda_visualization.html`!")
    else:
        st.warning(f"File visualisasi tidak ditemukan di `{vis_html_path}`.")
        st.info("Pastikan Anda sudah menyimpan file HTML dari pyLDAvis ke dalam folder `assets` Anda.")
        st.image("https://raw.githubusercontent.com/bmabey/pyLDAvis/master/notebooks/pyLDAvis_example.png",
                 caption="Contoh visualisasi pyLDAvis. File Anda akan ditampilkan di sini.")

with tab2:
    st.subheader("Rincian Topik yang Ditemukan")
    if 'topics' in lda_results:
        for topic_name, keywords in lda_results['topics'].items():
            st.markdown(f"**{topic_name}**")
            st.text(", ".join(keywords))
            st.markdown("---")
    else:
        st.warning("Tidak dapat menampilkan detail topik dari laporan.")

with tab3:
    st.subheader("Log Lengkap Proses Pemodelan LDA")
    st.code(lda_report_full, language="text")