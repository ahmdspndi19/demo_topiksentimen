# pages/5_Demo_&_Evaluasi.py
import streamlit as st
import time
import os
from utils import load_text_file # Impor fungsi untuk membaca file teks

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Demo & Evaluasi",
    page_icon="🚀",
    layout="wide"
)

# --- JUDUL DAN DESKRIPSI ---
st.title("🚀 Demo Interaktif & Evaluasi Kinerja Model")
st.markdown("Coba model analisis sentimen secara langsung dan lihat metrik kinerjanya berdasarkan laporan pelatihan.")
st.markdown("---")

# --- MEMUAT LAPORAN DARI FILE ---
# Path ke direktori laporan
REPORTS_DIR = "assets/reports"
report_sebelum = load_text_file(os.path.join(REPORTS_DIR, "report_sebelum.txt"))
report_sesudah = load_text_file(os.path.join(REPORTS_DIR, "report_sesudah.txt"))
log_sebelum = load_text_file(os.path.join(REPORTS_DIR, "training_log_sebelum.txt"))
log_sesudah = load_text_file(os.path.join(REPORTS_DIR, "training_log_sesudah.txt"))

# --- KONTEN HALAMAN ---
col1, col2 = st.columns(2)

# --- KOLOM MODEL SEBELUM ---
with col1:
    st.subheader("📉 Model Sebelum: LSTM Sederhana")
    image_path_before = "assets/image_6737bc.png"
    if os.path.exists(image_path_before):
        # --- PERBAIKAN DI SINI ---
        st.image(image_path_before, caption="Confusion Matrix Model Dasar", use_container_width=True)
    else:
        st.warning(f"File gambar tidak ditemukan: '{image_path_before}'.")

    st.metric("Akurasi", "67.5%", delta="-16.6%", delta_color="inverse")
    st.metric("F1-Score (Macro)", "26.9%", delta="-46.6%", delta_color="inverse")

    with st.expander("Lihat Detail Analisis Lengkap (dari file)"):
        st.warning("**Kelemahan Model:**")
        st.markdown("""
        - **Gagal Total:** Model ini sama sekali tidak bisa memprediksi sentimen 'Negatif' dan 'Netral', seperti yang terlihat dari nilai *precision*, *recall*, dan *f1-score* yang nol.
        - **Tidak Seimbang:** Akurasi 67.5% hanya mencerminkan kemampuan model menebak kelas mayoritas ('Positif'), bukan kemampuan klasifikasi yang sebenarnya.
        """)
        st.markdown("---")
        st.markdown("##### Laporan Klasifikasi Lengkap")
        st.code(report_sebelum, language='text')
        st.markdown("##### Riwayat Pelatihan (Epochs)")
        st.code(log_sebelum, language='text')


# --- KOLOM MODEL SESUDAH ---
with col2:
    st.subheader("📈 Model Sesudah: Stacked Bi-LSTM")
    image_path_after = "assets/image_673b45.png"
    if os.path.exists(image_path_after):
        # --- PERBAIKAN DI SINI ---
        st.image(image_path_after, caption="Confusion Matrix Model Optimal", use_container_width=True)
    else:
        st.warning(f"File gambar tidak ditemukan: '{image_path_after}'.")

    st.metric("Akurasi", "84.1%", delta="16.6%")
    st.metric("F1-Score (Macro)", "73.5%", delta="46.6%")
    
    with st.expander("Lihat Detail Analisis Lengkap (dari file)"):
        st.success("**Peningkatan Signifikan:**")
        st.markdown("""
        - **Mampu Mengklasifikasi:** Model berhasil memprediksi ketiga kelas sentimen dengan cukup baik, terutama untuk kelas 'Negatif' dan 'Positif'.
        - **Penanganan Kelas Minoritas:** Meskipun kelas 'Netral' masih menjadi tantangan (*f1-score* 50.8%), model ini menunjukkan peningkatan besar dari nol.
        - **Arsitektur Efektif:** Penggunaan `Bidirectional LSTM` dan `class_weight` terbukti efektif menangkap pola yang lebih kompleks dalam data.
        - **Early Stopping:** Pelatihan berhenti di epoch ke-7, mencegah *overfitting* dan menghemat waktu komputasi.
        """)
        st.markdown("---")
        st.markdown("##### Laporan Klasifikasi Lengkap")
        st.code(report_sesudah, language='text')
        st.markdown("##### Riwayat Pelatihan (Epochs)")
        st.code(log_sesudah, language='text')

st.markdown("---")
st.info("💡 **Demo Interaktif:** Coba masukkan kalimat ulasan di bawah untuk melihat bagaimana model idealnya akan bekerja.")

# --- DEMO INTERAKTIF ---
user_input = st.text_area("Masukkan teks ulasan di sini:", "Kualitas kameranya bagus sekali, tapi daya tahan baterainya kurang memuaskan.", height=100)

if st.button("Analisis Sentimen!"):
    if user_input:
        with st.spinner('Menganalisis...'):
            time.sleep(1) # Simulasi
        st.success("**Prediksi Sentimen: Positif** (dengan topik Negatif terdeteksi)")
        st.balloons()
    else:
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")