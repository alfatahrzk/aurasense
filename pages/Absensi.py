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

# Set page config with custom theme
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
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .content {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .stRadio > div {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="header"><h1>📸 Absensi Harian</h1></div>', unsafe_allow_html=True)

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

with st.markdown('<div class="content">', unsafe_allow_html=True):
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

    st.markdown("</div>", unsafe_allow_html=True)

# 2. PILIH TIPE ABSENSI
with st.markdown('<div class="content">', unsafe_allow_html=True):
    st.markdown("<h3 style='color: #003366;'>Pilih Jenis Absensi</h3>", unsafe_allow_html=True)
    absen_type = st.radio("", ["Masuk", "Keluar"], horizontal=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

if 'berhasil_absen' not in st.session_state:
    st.session_state['berhasil_absen'] = None

# --- UI LOGIC ---

if st.session_state['berhasil_absen'] is not None:
    user_data = st.session_state['berhasil_absen']
    
    with st.markdown('<div class="content">', unsafe_allow_html=True):
        st.markdown("<h3 style='color: #003366; text-align: center;'>✅ Absensi Berhasil!</h3>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #003366;'>
            <h4 style='color: #003366; text-align: center; margin-top: 0;'>STRUK BUKTI KEHADIRAN</h4>
            <hr style='border: 1px solid #003366;'>
            <p style='color: #003366;'><strong>Nama</strong>   : {user_data['nama']}</p>
            <p style='color: #003366;'><strong>Waktu</strong>  : {user_data['waktu']}</p>
            <p style='color: #003366;'><strong>Lokasi</strong> : {user_data.get('alamat', '-')}</p>
            <hr style='border: 1px solid #003366;'>
            <p style='color: #003366; text-align: center;'>Data tersimpan di Cloud.</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔄 Kembali ke Kamera", type="primary"):
        st.session_state['berhasil_absen'] = None 
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

else:
    with st.markdown('<div class="content">', unsafe_allow_html=True):
        st.markdown("<h3 style='color: #003366; text-align: center;'>Scan Wajah Anda</h3>", unsafe_allow_html=True)
        img_file = st.camera_input("", key="absen_cam")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        
        raw_cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
        cv_img = cv2.flip(raw_cv_img, 1)
        
        coords = engine.extract_face_coords(cv_img)
        
        if coords is None:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.warning("⚠️ Wajah tidak terdeteksi.")
            st.image(cv_img, channels="BGR", caption="Gagal Deteksi")
            st.markdown("</div>", unsafe_allow_html=True)
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
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.error(f"🔴 Ditolak! Kualitas foto buruk / Terindikasi Spoofing.")
                st.info("💡 Pastikan kamera anda bersih dan pencahayaan cukup")
                st.markdown("</div>", unsafe_allow_html=True)
                
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
    st.markdown("</div>", unsafe_allow_html=True)
