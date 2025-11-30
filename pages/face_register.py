# pages/1_Registrasi_Admin.py
import streamlit as st
import cv2
import numpy as np
from core.engines import FaceEngine
from core.database import VectorDB

st.set_page_config(page_title="Registrasi Wajah", layout="centered")

# --- LOGIN ADMIN SEDERHANA ---
# (Bisa dipercanggih nanti, untuk sekarang pakai password hardcode)
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False

if not st.session_state['is_admin']:
    st.title("🔒 Admin Login")
    pwd = st.text_input("Masukkan Password Admin", type="password")
    if st.button("Login"):
        if pwd == "admin123": # Ganti password sesuka hati
            st.session_state['is_admin'] = True
            st.rerun()
        else:
            st.error("Password Salah!")
    st.stop() # Stop eksekusi jika belum login

# --- MAIN APP ---
st.title("📝 Registrasi Wajah")

# Load Backend (Cached)
@st.cache_resource
def get_backends():
    return FaceEngine(), VectorDB()

engine, db = get_backends()

# Input Nama
username = st.text_input("Nama Karyawan Baru")

# State Management untuk 8 Pose
if 'reg_data' not in st.session_state:
    st.session_state['reg_data'] = [] # List menampung semua embedding
if 'step' not in st.session_state:
    st.session_state['step'] = 0

instructions = [
    "😐 1. Wajah Datar (Netral)",
    "😁 2. Tersenyum Lebar",
    "↗️ 3. Hadap Serong Kanan",
    "↖️ 4. Hadap Serong Kiri",
    "⬆️ 5. Menghadap Atas (Dagu Naik)",
    "⬇️ 6. Menghadap Bawah (Dagu Turun)",
    "🤪 7. Miringkan Kepala Kanan",
    "🤪 8. Miringkan Kepala Kiri"
]

total_steps = 8
current = st.session_state['step']

if current < total_steps:
    st.info(f"Langkah {current + 1}/{total_steps}: {instructions[current]}")
    
    # Key unik per step agar kamera refresh
    img_file = st.camera_input("Ambil Foto", key=f"cam_{current}")
    
    if img_file:
        bytes_data = img_file.getvalue()
        raw_cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)

        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
        
        # 1. Proses Gambar Normal
        face_normal = engine.detect_face(cv_img)
        
        if face_normal is None:
            st.warning("⚠️ Wajah tidak terdeteksi. Posisikan wajah di tengah.")
        else:
            # 2. Proses Augmentasi Gelap (Low Light)
            cv_img_dark = engine.simulate_low_light(cv_img, gamma=0.4)
            face_dark = engine.detect_face(cv_img_dark)
            
            if face_dark is None:
                st.warning("⚠️ Gagal deteksi pada simulasi gelap. Pastikan ruangan cukup terang.")
            else:
                # Sukses! Ekstrak kedua vektor
                emb1 = engine.get_embedding(face_normal)
                emb2 = engine.get_embedding(face_dark)
                
                # Simpan
                st.session_state['reg_data'].extend([emb1, emb2])
                st.session_state['step'] += 1
                st.toast(f"Pose {current+1} Tersimpan (+ Versi Gelap)", icon="✅")
                st.rerun()

    # Progress Bar
    st.progress(current / total_steps)

else:
    # FINISHING
    st.success("✅ Pengambilan Data Selesai!")
    st.write(f"Total Sampel Vektor: {len(st.session_state['reg_data'])} (8 Pose x 2 Kondisi Cahaya)")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Simpan ke Database", type="primary"):
            if not username:
                st.error("Nama wajib diisi!")
            else:
                with st.spinner("Menghitung Master Embedding & Upload ke Cloud..."):
                    # Hitung Rata-rata
                    master_emb = engine.calculate_average_embedding(st.session_state['reg_data'])
                    # Simpan ke Qdrant
                    db.save_user(username, master_emb)
                    
                    st.balloons()
                    st.success(f"Sukses! {username} telah terdaftar.")
                    
                    # Reset
                    st.session_state['reg_data'] = []
                    st.session_state['step'] = 0
                    st.rerun()
    
    with col2:
        if st.button("🔄 Ulangi dari Awal"):
            st.session_state['reg_data'] = []
            st.session_state['step'] = 0
            st.rerun()