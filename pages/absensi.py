import streamlit as st
import cv2
import numpy as np
from haversine import haversine, Unit
from datetime import datetime
import pandas as pd
import time

# --- PERBAIKAN 1: IMPORT YANG BENAR (face_engine) ---
from core.engines import FaceEngine 
from core.database import VectorDB
from core.logger import AttendanceLogger
from core.config_manager import ConfigManager
from core.locator import LocationService 

st.set_page_config(page_title="Absensi Karyawan", layout="centered")

# --- INISIALISASI BACKEND ---
@st.cache_resource
def get_backends():
    return FaceEngine(), VectorDB(), AttendanceLogger(), LocationService(), ConfigManager()

engine, db, logger, locator, config_mgr = get_backends()

# --- FUNGSI LOGIKA LOKASI ---
def check_location(user_lat, user_lon, office_lat, office_lon, radius_km):
    distance = haversine((user_lat, user_lon), (office_lat, office_lon), unit=Unit.KILOMETERS)
    return distance, distance <= radius_km

# --- HALAMAN UTAMA ---
st.title("📸 Absensi Harian")

# 1. DETEKSI LOKASI OTOMATIS
office_conf = config_mgr.get_config()
OFFICE_COORD = (office_conf['office_lat'], office_conf['office_lon'])
MAX_RADIUS_KM = office_conf['radius_km']

st.subheader("📍 Deteksi Lokasi")

with st.spinner("Mencari koordinat Anda..."):
    user_lat, user_lon, source = locator.get_coordinates()

# TAMPILKAN STATUS LOKASI
if user_lat is None:
    st.warning("⚠️ Sedang meminta izin lokasi browser... (Klik 'Allow' jika muncul pop-up)")
    st.info("Jika tidak muncul, pastikan GPS aktif. Sistem akan mencoba menggunakan IP Address.")
    st.stop() # Stop sampai lokasi ditemukan

else:
    # Hitung Jarak
    distance, is_in_radius = check_location(user_lat, user_lon, *OFFICE_COORD, MAX_RADIUS_KM)
    
    with st.spinner("Mendeteksi nama jalan..."):
        current_address = locator.get_address(user_lat, user_lon)
    
    # Tampilkan Alamat
    st.info(f"📍 Posisi anda di : **{current_address}**")
    
    # --- PERBAIKAN 2: TAMPILKAN ANGKA KOORDINAT ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Latitude", f"{user_lat:.4f}")
    col2.metric("Longitude", f"{user_lon:.4f}")
    col3.metric("Sumber", source)
    # ----------------------------------------------
    
    if is_in_radius:
        st.success(f"✅ Lokasi Valid! Jarak: {distance:.3f} km")
    else:
        st.error(f"❌ Di Luar Kantor! Jarak: {distance:.3f} km")
        st.stop()

st.divider()

# 2. PILIH TIPE ABSENSI
absen_type = st.radio("Jenis Absensi:", ["Masuk", "Keluar"], horizontal=True)

# --- STATE MANAGEMENT ---
if 'berhasil_absen' not in st.session_state:
    st.session_state['berhasil_absen'] = None

# --- LOGIKA UI: TAMPILKAN HASIL ATAU KAMERA ---

if st.session_state['berhasil_absen'] is not None:
    user_data = st.session_state['berhasil_absen']
    
    st.balloons() 
    st.success(f"✅ Absensi Berhasil!")
    
    st.info(f"""
    **STRUK BUKTI KEHADIRAN**
    -------------------------
    👤 Nama   : {user_data['nama']}
    📅 Waktu  : {user_data['waktu']}
    📍 Jarak  : {user_data['jarak']} km
    🏠 Alamat : {user_data.get('alamat', '-')}
    -------------------------
    Data telah tersimpan di Cloud Database.
    """)
    
    if st.button("🔄 Kembali ke Kamera", type="primary"):
        st.session_state['berhasil_absen'] = None 
        st.rerun()

else:
    # 3. KAMERA INPUT
    img_file = st.camera_input("Scan Wajah Anda", key="absen_cam")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        
        # Decode & Mirroring
        raw_cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
        cv_img = cv2.flip(raw_cv_img, 1) # Mirroring Aktif
        
        # Gunakan MediaPipe (extract_face_coords)
        coords = engine.extract_face_coords(cv_img)
        
        if coords is None:
            st.warning("⚠️ Wajah tidak terdeteksi.")
            st.image(cv_img, channels="BGR", caption="Gagal Deteksi")
        else:
            x, y, w, h = coords 
            
            # Gambar Kotak
            img_with_box = cv_img.copy()
            cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
            st.image(img_with_box, channels="BGR", caption="Wajah Terdeteksi")
            
            # Crop Wajah
            face_crop = cv_img[y:y+h, x:x+w]
            
            with st.spinner("Mencocokkan biometrik..."):
                input_emb = engine.get_embedding(face_crop)
                
                # Threshold 0.7 sudah cukup ketat dan aman
                found_user, score = db.search_user(input_emb, threshold=0.65)
                
                if found_user:
                    if logger.log_attendance(found_user, absen_type, distance, current_address):
                        
                        st.session_state['berhasil_absen'] = {
                            'nama': found_user,
                            'waktu': datetime.now().strftime('%H:%M:%S'),
                            'jarak': f"{distance:.3f}",
                            'alamat': current_address
                        }
                        st.rerun()
                    else:
                        st.error("Gagal terhubung ke Database Log.")
                else:
                    st.error("❌ Wajah tidak dikenali! Silakan coba lagi.")