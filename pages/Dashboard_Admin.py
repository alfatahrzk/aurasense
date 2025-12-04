import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

from core.engines import FaceEngine 
from core.database import VectorDB
from core.config_manager import ConfigManager 
from core.logger import AttendanceLogger 
from core.admin_auth import AdminAuth # <--- IMPORT BARU

st.set_page_config(page_title="Dashboard Admin", layout="wide") 

# Custom CSS agar selaras dengan home.py dan Absensi.py
st.markdown("""
    <style>
        .main {
            background-color: #e6f2ff;
        }
        .stApp {
            background-color: #e6f2ff;
            color: #003366;
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
        /* Tab styling agar teks tidak menyatu dengan background */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #ffffff;
            color: #003366;
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #003366 !important;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.1);
        }
        /* Make specific headers and labels dark blue */
        .stHeader {
            color: #003366 !important;
        }
        /* Custom class for dark blue text */
        .dark-blue-text {
            color: #003366 !important;
            font-weight: 600;
        }
        .stTextInput > div > div > input,
        .stTextInput > div > label {
            color: #003366 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- INISIALISASI ---
@st.cache_resource
def get_backends():
    return FaceEngine(), VectorDB(), ConfigManager(), AttendanceLogger(), AdminAuth()

engine, db, config_mgr, logger, auth = get_backends()

# --- LOGIN ADMIN (VERSI DATABASE) ---
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False
if 'admin_name' not in st.session_state:
    st.session_state['admin_name'] = ""

if not st.session_state['is_admin']:
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.title("🔒 Admin Login System")
        
        form_user = st.text_input("Username")
        form_pass = st.text_input("Password", type="password")
        
        if st.button("Masuk", type="primary", use_container_width=True):
            if auth.login(form_user, form_pass):
                st.session_state['is_admin'] = True
                st.session_state['admin_name'] = form_user
                st.toast(f"Selamat datang, {form_user}!", icon="👋")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Username atau Password salah!")
    st.stop() 

# --- SIDEBAR LOGOUT ---
with st.sidebar:
    st.write(f"Login sebagai: **{st.session_state['admin_name']}**")
    if st.button("Logout"):
        st.session_state['is_admin'] = False
        st.rerun()

# Header utama selaras dengan halaman lain
st.markdown('<div class="header"><h1>⚙️ Dashboard Admin</h1></div>', unsafe_allow_html=True)

# BUAT 5 TAB MENU (TAMBAHAN SATU TAB BARU)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Registrasi Wajah", 
    "🎛️ Pengaturan Sistem", 
    "📊 Riwayat Absensi",
    "👥 Kelola Wajah",
    "🔑 Kelola Akun Admin" # <--- TAB BARU
])

# ====================================================
# TAB 1: REGISTRASI WAJAH (SAMA)
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
# TAB 2: PENGATURAN SISTEM (SAMA)
# ====================================================
with tab2:
    st.header("Konfigurasi Global")
    @st.cache_data(ttl=10)
    def load_config(): return config_mgr.get_config()
    current_conf = load_config()
    
    with st.form("edit_config"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="dark-blue-text">📍 Lokasi Kantor</p>', unsafe_allow_html=True)
            st.markdown('<p class="dark-blue-text">Latitude</p>', unsafe_allow_html=True)
            lat = st.number_input("", value=float(current_conf.get('office_lat', -7.25)), format="%.6f", label_visibility="collapsed")
            st.markdown('<p class="dark-blue-text">Longitude</p>', unsafe_allow_html=True)
            lon = st.number_input("", value=float(current_conf.get('office_lon', 112.75)), format="%.6f", label_visibility="collapsed")
            st.markdown('<p class="dark-blue-text">Radius (km)</p>', unsafe_allow_html=True)
            rad = st.number_input("", value=float(current_conf.get('radius_km', 0.5)), step=0.1, label_visibility="collapsed")
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
# TAB 3: RIWAYAT ABSENSI (SAMA)
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
# TAB 4: KELOLA WAJAH (SAMA)
# ====================================================
with tab4:
    st.header("👥 Database Wajah (Qdrant)")
    
    with st.spinner("Mengambil daftar karyawan..."):
        users_list = db.get_all_users()
    
    if not users_list:
        st.warning("Database Qdrant masih kosong.")
    else:
        df_users = pd.DataFrame(users_list, columns=["Nama Karyawan Terdaftar"])
        df_users.index = df_users.index + 1
        
        col_info, col_table = st.columns([1, 2])
        with col_info:
            st.success(f"Total: {len(users_list)} Karyawan")
        with col_table:
            st.dataframe(df_users, use_container_width=True)

        st.divider()
        st.subheader("🗑️ Hapus Data Karyawan")
        
        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            user_to_delete = st.selectbox("Pilih nama karyawan:", users_list)
        with col_del2:
            st.write("") 
            st.write("") 
            if st.button("Hapus Permanen", type="primary"):
                if user_to_delete:
                    with st.spinner(f"Menghapus data {user_to_delete}..."):
                        if db.delete_user(user_to_delete):
                            st.toast(f"User {user_to_delete} berhasil dihapus!", icon="🗑️")
                            time.sleep(1)
                            st.rerun() 
                        else:
                            st.error("Gagal menghapus user.")

# ====================================================
# TAB 5: KELOLA ADMIN (FITUR BARU)
# ====================================================
with tab5:
    st.header("🔑 Manajemen Akun Admin")
    
    # Bagian 1: List Admin
    st.subheader("Daftar Admin Terdaftar")
    admins = auth.get_all_admins()
    
    if admins:
        df_admin = pd.DataFrame(admins)
        st.dataframe(df_admin, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Bagian 2: Tambah Admin Baru
    col_add, col_rem = st.columns(2)
    
    with col_add:
        st.subheader("➕ Tambah Admin Baru")
        with st.form("add_admin_form"):
            new_user = st.text_input("Username Baru")
            new_pass = st.text_input("Password Baru", type="password")
            submitted = st.form_submit_button("Tambah Admin")
            
            if submitted:
                if new_user and new_pass:
                    success, msg = auth.add_admin(new_user, new_pass)
                    if success:
                        st.success(f"Admin {new_user} berhasil ditambahkan!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Isi username dan password.")
    
    # Bagian 3: Hapus Admin
    with col_rem:
        st.subheader("⛔ Hapus Admin")
        
        # Ambil list username saja
        admin_usernames = [a['username'] for a in admins] if admins else []
        
        # Jangan izinkan hapus diri sendiri (cegah bunuh diri akun)
        current_user = st.session_state.get('admin_name', '')
        valid_to_delete = [u for u in admin_usernames if u != current_user]
        
        if valid_to_delete:
            del_target = st.selectbox("Pilih Admin untuk dihapus:", valid_to_delete)
            
            if st.button("Hapus Admin Terpilih", type="primary"):
                if auth.delete_admin(del_target):
                    st.success(f"Admin {del_target} dihapus.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Gagal menghapus.")
        else:
            st.info("Tidak ada admin lain yang bisa dihapus (Anda tidak bisa menghapus diri sendiri).")
