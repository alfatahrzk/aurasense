import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

# Import Backend
from core.engines import FaceEngine 
from core.database import VectorDB
from core.config_manager import ConfigManager # <--- Jangan lupa ini

st.set_page_config(page_title="Dashboard Admin", layout="centered")

# --- LOGIN ADMIN ---
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False

if not st.session_state['is_admin']:
    st.title("🔒 Admin Login")
    pwd = st.text_input("Masukkan Password Admin", type="password")
    if st.button("Login"):
        correct_pwd = st.secrets.get("ADMIN_PASSWORD", "admin123")
        if pwd == correct_pwd: 
            st.session_state['is_admin'] = True
            st.rerun()
        else:
            st.error("Password Salah!")
    st.stop() 

# --- INISIALISASI ---
@st.cache_resource
def get_backends():
    return FaceEngine(), VectorDB(), ConfigManager()

engine, db, config_mgr = get_backends()

# --- HEADER ---
st.title("⚙️ Dashboard Admin")

# BUAT 2 TAB MENU
tab1, tab2 = st.tabs(["📝 Registrasi Wajah", "📍 Lokasi Kantor"])

# ====================================================
# TAB 1: REGISTRASI WAJAH (Kode Anda yang sudah benar)
# ====================================================
with tab1:
    st.header("Pendaftaran Karyawan Baru")
    
    username = st.text_input("Nama Karyawan Baru")

    if 'reg_data' not in st.session_state:
        st.session_state['reg_data'] = [] 
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
        
        # Key unik
        img_file = st.camera_input("Ambil Foto", key=f"cam_{current}")
        
        if img_file:
            bytes_data = img_file.getvalue()
            
            # Decode & Mirroring
            raw_cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
            cv_img = cv2.flip(raw_cv_img, 1)
            
            # Deteksi
            coords = engine.extract_face_coords(cv_img)
            
            if coords is None:
                st.warning("⚠️ Wajah tidak terdeteksi.")
                st.image(cv_img, channels="BGR", caption="Gagal Deteksi")
            else:
                x, y, w, h = coords
                
                # Visualisasi
                img_with_box = cv_img.copy()
                cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
                st.image(img_with_box, channels="BGR", caption=f"Pose {current+1} Oke")
                
                # Crop & Embed
                face_crop = cv_img[y:y+h, x:x+w]
                emb = engine.get_embedding(face_crop)
                
                st.session_state['reg_data'].append(emb)
                st.session_state['step'] += 1
                st.toast(f"Pose {current+1} Tersimpan", icon="✅")
                
                time.sleep(1)
                st.rerun()

        st.progress(current / total_steps)

    else:
        st.success("✅ Data Lengkap!")
        st.write(f"Sampel terkumpul: {len(st.session_state['reg_data'])}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Simpan ke Database", type="primary"):
                if not username:
                    st.error("Nama wajib diisi!")
                else:
                    with st.spinner("Upload ke Qdrant..."):
                        master_emb = engine.calculate_average_embedding(st.session_state['reg_data'])
                        success = db.save_user(username, master_emb)
                        
                        if success:
                            st.balloons()
                            st.success(f"Sukses! {username} terdaftar.")
                            st.session_state['reg_data'] = []
                            st.session_state['step'] = 0
                            st.rerun()
                        else:
                            st.error("Gagal simpan.")
        with col2:
            if st.button("🔄 Reset"):
                st.session_state['reg_data'] = []
                st.session_state['step'] = 0
                st.rerun()

# ====================================================
# TAB 2: PENGATURAN LOKASI (Agar Admin Bisa Ubah)
# ====================================================
with tab2:
    st.header("Lokasi Kantor & Radius")
    
    # Fungsi Load Cache
    @st.cache_data(ttl=60)
    def load_config():
        return config_mgr.get_config()
    
    current_conf = load_config()
    
    with st.form("edit_lokasi"):
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", value=float(current_conf.get('office_lat', -7.25)), format="%.6f")
        with c2:
            lon = st.number_input("Longitude", value=float(current_conf.get('office_lon', 112.75)), format="%.6f")
            
        rad = st.number_input("Radius (km)", value=float(current_conf.get('radius_km', 0.5)), step=0.1)
        
        if st.form_submit_button("Simpan Konfigurasi"):
            if config_mgr.save_config(lat, lon, rad):
                load_config.clear() # Hapus cache biar update
                st.success("Tersimpan!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Gagal update config.")