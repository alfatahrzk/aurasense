import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

from core.engines import FaceEngine 
from core.database import VectorDB
from core.config_manager import ConfigManager 
from core.logger import AttendanceLogger 

st.set_page_config(page_title="Dashboard Admin", layout="wide") 

# --- LOGIN ADMIN ---
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False

if not st.session_state['is_admin']:
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.title("🔒 Admin Login")
        pwd = st.text_input("Masukkan Password Admin", type="password")
        if st.button("Login", use_container_width=True):
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
    return FaceEngine(), VectorDB(), ConfigManager(), AttendanceLogger()

engine, db, config_mgr, logger = get_backends()

st.title("⚙️ Dashboard Admin")

# BUAT 4 TAB MENU (TAMBAHAN SATU TAB BARU)
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Registrasi Wajah", 
    "🎛️ Pengaturan Sistem", 
    "📊 Riwayat Absensi",
    "👥 Kelola Wajah"
])

# ====================================================
# TAB 1: REGISTRASI WAJAH
# ====================================================
with tab1:
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
                if st.button("🔄 Ulangi", use_container_width=True):
                    st.session_state['reg_data'] = []
                    st.session_state['step'] = 0
                    st.rerun()

# ====================================================
# TAB 2: PENGATURAN SISTEM
# ====================================================
with tab2:
    st.header("Konfigurasi Global")
    @st.cache_data(ttl=10)
    def load_config(): return config_mgr.get_config()
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
        
        if st.form_submit_button("Simpan Konfigurasi", use_container_width=True):
            if config_mgr.save_config(lat, lon, rad, face_thresh, liveness_thresh):
                load_config.clear()
                st.success("✅ Tersimpan!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Gagal update.")

# ====================================================
# TAB 3: RIWAYAT ABSENSI
# ====================================================
with tab3:
    st.header("📊 Data Log Absensi")
    if st.button("🔄 Refresh Log"):
        st.cache_data.clear()
        st.rerun()
    
    df_logs = logger.get_logs(limit=100)
    if not df_logs.empty:
        st.dataframe(df_logs, use_container_width=True, hide_index=True)
        csv = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="logs.csv", mime='text/csv')
    else:
        st.info("Belum ada data.")

# ====================================================
# TAB 4: KELOLA WAJAH (TAMPILAN TABEL)
# ====================================================
with tab4:
    st.header("👥 Database Wajah (Qdrant)")
    
    # 1. Ambil Data dari Qdrant
    with st.spinner("Mengambil daftar karyawan..."):
        users_list = db.get_all_users()
    
    if not users_list:
        st.warning("Database Qdrant masih kosong.")
    else:
        # --- PERBAIKAN TAMPILAN: UBAH LIST JADI TABEL ---
        
        # Buat DataFrame
        df_users = pd.DataFrame(users_list, columns=["Nama Karyawan Terdaftar"])
        
        # Tambahkan kolom Nomor (Index + 1)
        df_users.index = df_users.index + 1
        df_users.index.name = "No"
        
        col_info, col_table = st.columns([1, 2])
        
        with col_info:
            st.success(f"Total: {len(users_list)} Karyawan")
            st.info("Daftar ini diambil langsung dari Vector Database Cloud (Qdrant).")
            
        with col_table:
            # Tampilkan Tabel
            st.dataframe(df_users, use_container_width=True)

        st.divider()
        
        # FITUR HAPUS USER (Tetap pakai Dropdown agar aman)
        st.subheader("🗑️ Hapus Data Karyawan")
        st.caption("Peringatan: Data yang dihapus tidak bisa dikembalikan.")
        
        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            user_to_delete = st.selectbox("Pilih nama karyawan yang akan dihapus:", users_list)
        
        with col_del2:
            st.write("") # Spacer agar tombol sejajar ke bawah
            st.write("") 
            if st.button("Hapus Permanen", type="primary"):
                if user_to_delete:
                    with st.spinner(f"Menghapus data {user_to_delete}..."):
                        success = db.delete_user(user_to_delete)
                        if success:
                            st.toast(f"User {user_to_delete} berhasil dihapus!", icon="🗑️")
                            time.sleep(1)
                            st.rerun() # Refresh halaman agar tabel update
                        else:
                            st.error("Gagal menghapus user.")