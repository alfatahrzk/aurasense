import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime


from core.engines import FaceEngine 
from core.database import VectorDB
from core.config_manager import ConfigManager 
from core.logger import AttendanceLogger 

# Gunakan layout wide agar tabel log nanti lega, tapi form registrasi kita tengahkan manual
st.set_page_config(page_title="Dashboard Admin", layout="wide") 

# --- LOGIN ADMIN ---
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False

if not st.session_state['is_admin']:
    # Tampilan Login Ditengahkan juga
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.title("🔒 Admin Login")
        pwd = st.text_input("Masukkan Password Admin", type="password")
        if st.button("Login", use_container_width=True):
            correct_pwd = st.secrets.get("ADMIN_PASSWORD", "kopihitamnyamandilambung")
            if pwd == correct_pwd: 
                st.session_state['is_admin'] = True
                st.rerun()
            else:
                st.error("Password Salah!")
    st.stop() 

# --- INISIALISASI ---
@st.cache_resource
def get_backends():
    return FaceEngine(), VectorDB(), ConfigManager(), AttendanceLogger()

engine, db, config_mgr, logger = get_backends()

st.title("⚙️ Dashboard Admin")

# BUAT 3 TAB MENU
tab1, tab2, tab3 = st.tabs(["📝 Registrasi Wajah", "🎛️ Pengaturan Sistem", "📊 Riwayat Absensi"])

# ====================================================
# TAB 1: REGISTRASI WAJAH (SUDAH DITENGAHKAN)
# ====================================================
with tab1:
    # --- PERBAIKAN TAMPILAN DI SINI ---
    # Kita bagi layar jadi 3: [1 bagian kosong, 2 bagian isi, 1 bagian kosong]
    # Agar form ada di tengah
    c_left, c_center, c_right = st.columns([1, 2, 1])
    
    with c_center:
        st.header("Pendaftaran Karyawan")
        username = st.text_input("Nama Karyawan Baru")

        if 'reg_data' not in st.session_state: st.session_state['reg_data'] = [] 
        if 'step' not in st.session_state: st.session_state['step'] = 0

        instructions = [
            "😐 1. Wajah Datar (Netral)", "😁 2. Tersenyum Lebar",
            "↗️ 3. Hadap Serong Kanan", "↖️ 4. Hadap Serong Kiri",
            "⬆️ 5. Menghadap Atas", "⬇️ 6. Menghadap Bawah",
            "🤪 7. Miring Kanan", "🤪 8. Miring Kiri"
        ]
        total_steps = 8
        current = st.session_state['step']

        if current < total_steps:
            # Tampilkan instruksi dengan alert biru agar jelas
            st.info(f"**Langkah {current + 1}/{total_steps}:** {instructions[current]}")
            
            img_file = st.camera_input("Ambil Foto", key=f"cam_{current}")
            
            if img_file:
                bytes_data = img_file.getvalue()
                raw_cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
                cv_img = cv2.flip(raw_cv_img, 1)
                
                coords = engine.extract_face_coords(cv_img)
                
                if coords is None:
                    st.warning("⚠️ Wajah tidak terdeteksi.")
                else:
                    x, y, w, h = coords
                    img_box = cv_img.copy()
                    cv2.rectangle(img_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    face_crop = cv_img[y:y+h, x:x+w]
                    emb = engine.get_embedding(face_crop)
                    
                    st.session_state['reg_data'].append(emb)
                    st.session_state['step'] += 1
                    st.toast(f"Pose {current+1} Tersimpan", icon="✅")
                    time.sleep(0.5)
                    st.rerun()
            
            st.progress(current / total_steps)

        else:
            st.success("✅ Data Lengkap!")
            
            # Tombol Simpan & Reset kita sejajarkan
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("💾 Simpan ke Database", type="primary", use_container_width=True):
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
            with btn_col2:
                if st.button("🔄 Reset / Ulangi", use_container_width=True):
                    st.session_state['reg_data'] = []
                    st.session_state['step'] = 0
                    st.rerun()

# ====================================================
# TAB 2: PENGATURAN SISTEM
# ====================================================
with tab2:
    st.header("Konfigurasi Global")
    
    @st.cache_data(ttl=10)
    def load_config():
        return config_mgr.get_config()
    
    current_conf = load_config()
    
    with st.form("edit_config"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 Lokasi Kantor")
            lat = st.number_input("Latitude", value=float(current_conf.get('office_lat', -7.25)), format="%.6f")
            lon = st.number_input("Longitude", value=float(current_conf.get('office_lon', 112.75)), format="%.6f")
            rad = st.number_input("Radius (km)", value=float(current_conf.get('radius_km', 0.5)), step=0.1)
        
        with col2:
            st.subheader("🧠 Sensitivitas AI")
            face_thresh = st.slider("Threshold Wajah", 0.0, 1.0, float(current_conf.get('face_threshold', 0.70)), 0.01)
            liveness_thresh = st.slider("Threshold Liveness", 0.0, 200.0, float(current_conf.get('liveness_threshold', 60.0)), 10.0)
        
        if st.form_submit_button("Simpan Semua Konfigurasi", use_container_width=True):
            if config_mgr.save_config(lat, lon, rad, face_thresh, liveness_thresh):
                load_config.clear()
                st.success("✅ Konfigurasi berhasil disimpan!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Gagal update config.")

# ====================================================
# TAB 3: RIWAYAT ABSENSI (FITUR BARU)
# ====================================================
with tab3:
    st.header("📊 Data Absensi Masuk & Keluar")
    
    col_ref, col_down = st.columns([1, 5])
    with col_ref:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Fungsi Load Logs
    def load_logs_data():
        return logger.get_logs(limit=100) # Ambil 100 data terakhir
    
    df_logs = load_logs_data()
    
    if not df_logs.empty:
        # Rapikan Tampilan Tabel
        # Pilih kolom yang mau ditampilkan saja agar rapi
        columns_to_show = ["waktu_absen", "nama", "status", "jarak", "alamat", "skor_kemiripan", "skor_liveness", "status_validasi"]
        
        # Filter kolom yang ada saja (biar gak error kalau kolom baru belum ada di DB lama)
        valid_cols = [c for c in columns_to_show if c in df_logs.columns]
        df_show = df_logs[valid_cols]
        
        # Rename kolom biar bahasa Indonesia yang bagus
        df_show = df_show.rename(columns={
            "waktu_absen": "Waktu",
            "nama": "Nama Karyawan",
            "status": "Tipe",
            "jarak": "Jarak",
            "alamat": "Alamat",
            "skor_kemiripan": "Skor AI",
            "skor_liveness": "Skor Anti Spoofing",
            "status_validasi": "Validasi"
        })
        
        # Tampilkan Tabel Interaktif
        st.dataframe(
            df_show, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Skor AI": st.column_config.ProgressColumn(
                    "Kecocokan Wajah",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "Validasi": st.column_config.TextColumn(
                    "Status Validasi",
                    help="Status apakah absensi diterima atau ditolak sistem"
                )
            }
        )
        
        # Tombol Download CSV
        csv = df_show.to_csv(index=False).encode('utf-8')
        with col_down:
            st.download_button(
                label="📥 Download Laporan (CSV)",
                data=csv,
                file_name=f"laporan_absensi_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )
            
    else:
        st.info("Belum ada data absensi yang terekam.")