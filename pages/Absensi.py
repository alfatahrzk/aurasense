import streamlit as st
import cv2
import numpy as np
from haversine import haversine, Unit
from datetime import datetime
import pandas as pd
import time

# --- IMPORT LIBRARY ---
from core.engines import FaceEngine 
from core.database import VectorDB
from core.logger import AttendanceLogger
from core.config_manager import ConfigManager
from core.locator import LocationService 

st.set_page_config(
    page_title="Absensi Karyawan",
    layout="centered",
    page_icon="📸"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #e6f2ff;
        }
        .stApp {
            background-color: #e6f2ff;
        }
        .header {
            background-color: #003366;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .content-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        /* Style for radio buttons */
        .stRadio > div {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
        }
        .stRadio > div > label {
            color: #000000 !important;
            font-weight: 600;
            font-size: 16px;
        }
        /* Style for selected radio button */
        .stRadio > div[data-baseweb='radio'] > div:first-child > div:first-child > div {
            background-color: #003366 !important;
            border-color: #003366 !important;
        }
        /* Style for success message */
        .stAlert {
            border-radius: 10px;
        }
        /* Style for success message (location valid) */
        .stAlert [data-testid='stMarkdownContainer'] p {
            color: #004d00 !important;
            font-weight: 500;
        }
        .stAlert [data-testid='stNotification'] {
            background-color: #e6ffe6 !important;
            border-left: 4px solid #006600 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="header"><h1 style="color: white; margin: 0;">📸 Absensi Harian</h1></div>', unsafe_allow_html=True)

# --- INISIALISASI BACKEND ---
@st.cache_resource
def get_backends():
    return FaceEngine(), VectorDB(), AttendanceLogger(), LocationService(), ConfigManager()

engine, db, logger, locator, config_mgr = get_backends()

# --- LOAD CONFIG GLOBAL ---
office_conf = config_mgr.get_config()
OFFICE_COORD = (float(office_conf.get('office_lat', -7.25)), float(office_conf.get('office_lon', 112.75)))
MAX_RADIUS_KM = float(office_conf.get('radius_km', 0.5))

# Settingan AI
THRESHOLD_VAL = float(office_conf.get('face_threshold', 0.70))
LIVENESS_VAL = float(office_conf.get('liveness_threshold', 60.0))

# --- FUNGSI LOKASI ---
def check_location(user_lat, user_lon, office_lat, office_lon, radius_km):
    distance = haversine((user_lat, user_lon), (office_lat, office_lon), unit=Unit.KILOMETERS)
    return distance, distance <= radius_km

# --- HALAMAN UTAMA ---
# Title is now in the header

with st.spinner("Mencari lokasi Anda..."):
    user_lat, user_lon, source = locator.get_coordinates()

# VALIDASI LOKASI
if user_lat is None:
    st.warning("⚠️ Sedang meminta izin lokasi browser...")
    st.stop()
else:
    distance, is_in_radius = check_location(user_lat, user_lon, *OFFICE_COORD, MAX_RADIUS_KM)
    
    with st.spinner("Mendeteksi nama jalan..."):
        current_address = locator.get_address(user_lat, user_lon)
    
    if is_in_radius:
        st.success(f"✅ Lokasi Valid! ({distance:.3f} km)")
    else:
        st.error(f"❌ Di Luar Kantor! Jarak: {distance:.3f} km")
        st.stop()

st.divider()

# 2. PILIH TIPE ABSENSI
st.markdown("<h3 style='color: #003366; margin-bottom: 10px;'>Pilih Jenis Absensi</h3>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    absen_type = st.radio("", ["Masuk", "Keluar"], horizontal=True, label_visibility="collapsed")

if 'berhasil_absen' not in st.session_state:
    st.session_state['berhasil_absen'] = None

# --- UI LOGIC ---

if st.session_state['berhasil_absen'] is not None:
    user_data = st.session_state['berhasil_absen']
    
    with st.container():
        # Ekstrak nilai dari user_data terlebih dahulu
        nama = user_data.get('nama', '-')
        waktu = user_data.get('waktu', '-')
        alamat = user_data.get('alamat', '-')
        
        st.markdown(f"""
        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;'>
            <h3 style='color: #28a745; text-align: center; margin-top: 0;'>✅ Absensi Berhasil!</h3>
            <div style='background-color: white; padding: 15px; border-radius: 8px;'>
                <h4 style='color: #003366; text-align: center; margin-top: 0;'>STRUK BUKTI KEHADIRAN</h4>
                <hr style='border: 1px solid #003366; opacity: 0.3;'>
                <p style='color: #003366;'><strong>Nama</strong>   : {nama}</p>
                <p style='color: #003366;'><strong>Waktu</strong>  : {waktu}</p>
                <p style='color: #003366;'><strong>Lokasi</strong> : {alamat}</p>
                <hr style='border: 1px solid #003366; opacity: 0.3;'>
                <p style='color: #003366; text-align: center; margin-bottom: 0;'>Data tersimpan di Cloud.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔄 Kembali ke Kamera", type="primary"):
        st.session_state['berhasil_absen'] = None 
        st.rerun()

else:
    with st.container():
        st.markdown("<h3 style='color: #003366; text-align: center;'>Scan Wajah Anda</h3>", unsafe_allow_html=True)
        img_file = st.camera_input("", key="absen_cam")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        
        raw_cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
        cv_img = cv2.flip(raw_cv_img, 1)
        
        coords = engine.extract_face_coords(cv_img)
        
        if coords is None:
            st.warning("⚠️ Wajah tidak terdeteksi.")
            st.image(cv_img, channels="BGR", caption="Gagal Deteksi")
        else:
            x, y, w, h = coords 
            face_crop = cv_img[y:y+h, x:x+w]

            # --- CEK LIVENESS ---
            _, liveness_score = engine.check_liveness(face_crop)
            is_real = liveness_score > LIVENESS_VAL
            
            if is_real:
                with st.spinner("Mencocokkan biometrik..."):
                    input_emb = engine.get_embedding(face_crop)
                    found_user, score = db.search_user(input_emb, threshold=0.0)
                                        
                    # --- CEK THRESHOLD WAJAH ---
                    if found_user and score >= THRESHOLD_VAL:
                        # --- KASUS SUKSES ---
                        sukses = logger.log_attendance(
                            name=found_user, 
                            status=absen_type, 
                            location_dist=distance, 
                            address=current_address,
                            lat=user_lat,
                            lon=user_lon,
                            similarity=score,
                            liveness=liveness_score,
                            validation_status="Berhasil"
                        )
                        
                        if sukses:
                            st.session_state['berhasil_absen'] = {
                                'nama': found_user,
                                'skor': f"{score:.4f}",
                                'waktu': (datetime.now() + pd.Timedelta(hours=7)).strftime('%H:%M:%S'), 
                                'jarak': f"{distance:.3f}",
                                'alamat': current_address
                            }
                            st.rerun()
                        else:
                            st.error("Gagal terhubung ke Database Log.")
                    else:
                        # --- KASUS GAGAL: SKOR RENDAH (TAPI LOGGING JALAN) ---
                        st.error(f"❌ Ditolak! Wajah tidak dikenali.\nHarap hubungi admin!")
                        
                        # Simpan Log Gagal
                        logger.log_attendance(
                            name=f"{found_user} (Ditolak)", # Catat siapa yang paling mirip
                            status="Gagal",
                            location_dist=distance,
                            address=current_address,
                            lat=user_lat,
                            lon=user_lon,
                            similarity=score,
                            liveness=liveness_score,
                            validation_status="Gagal: Skor Rendah"
                        )
            else:
                # --- KASUS GAGAL: SPOOFING (LOGGING JALAN) ---
                st.error(f"🔴 Ditolak! Kualitas foto buruk / Terindikasi Spoofing.")
                st.info("💡 Pastikan kamera anda bersih dan pencahayaan cukup")
                
                # Simpan Log Gagal
                logger.log_attendance(
                    name="Unknown (Spoof)",
                    status="Gagal",
                    location_dist=distance,
                    address=current_address,
                    lat=user_lat,
                    lon=user_lon,
                    similarity=0.0,
                    liveness=liveness_score,
                    validation_status="Gagal: Liveness Check"
                )
