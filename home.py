# Home.py
import streamlit as st

st.set_page_config(
    page_title="AuraSense",
    layout="centered",
)

st.title("🏢 AuraSense Presence")
st.image("https://cdn-icons-png.flaticon.com/512/3652/3652191.png", width=150)

st.markdown("""
### Selamat Datang
Sistem ini menggunakan teknologi **Face Recognition berbasis AI (ResNet50)** dengan penyimpanan **Vector Database (Qdrant)**.

Silakan pilih menu di sidebar (sebelah kiri):
* **Registrasi Wajah:** (Khusus Admin) Untuk mendaftarkan karyawan baru dengan 8 pose.
* **Absensi User:** (Akan dibuat selanjutnya) Untuk melakukan presensi harian.
""")

st.info("💡 Pastikan Anda memiliki akses internet stabil untuk terhubung ke Cloud Database.")