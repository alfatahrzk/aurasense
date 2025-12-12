import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import folium 
from streamlit_folium import st_folium 
from folium.plugins import Draw # <--- SENJATA RAHASIA KITA

from core.engines import FaceEngine 
from core.database import VectorDB
from core.config_manager import ConfigManager 
from core.logger import AttendanceLogger 
from core.admin_auth import AdminAuth 

st.set_page_config(page_title="Dashboard Admin", layout="wide") 

# --- CSS (SAMA) ---
st.markdown("""
    <style>
        .main { background-color: #e6f2ff; }
        .stApp { background-color: #e6f2ff; color: #003366; }
        .header { background-color: #003366; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { background-color: #ffffff; color: #003366; border-radius: 8px 8px 0 0; padding: 8px 16px; font-weight: 600; }
        .stTabs [aria-selected="true"] { background-color: #ffffff !important; color: #003366 !important; box-shadow: 0 -2px 4px rgba(0,0,0,0.1); }
        .dark-blue-text { color: #003366 !important; font-weight: 600; margin-bottom: 0.5rem; }
        .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button, button[data-testid="stBaseButton-secondaryFormSubmit"] { background-color: #003366 !important; color: white !important; }
        .st-emotion-cache-zuyloh.emjbblw1[data-testid="stForm"] { padding: 0 !important; }
        [data-testid="stForm"] [data-testid="stNumberInput"] label { color: #003366 !important; }
    </style>
""", unsafe_allow_html=True)

# --- INISIALISASI ---
@st.cache_resource
def get_backends():
    return FaceEngine(), VectorDB(), ConfigManager(), AttendanceLogger(), AdminAuth()

engine, db, config_mgr, logger, auth = get_backends()

# --- LOGIN ADMIN ---
if 'is_admin' not in st.session_state: st.session_state['is_admin'] = False
if 'admin_name' not in st.session_state: st.session_state['admin_name'] = ""

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
                st.rerun()
            else:
                st.error("Login Gagal!")
    st.stop() 

with st.sidebar:
    st.write(f"Login: **{st.session_state['admin_name']}**")
    if st.button("Logout"):
        st.session_state['is_admin'] = False
        st.rerun()

st.markdown('<div class="header"><h1>⚙️ Dashboard Admin</h1></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Registrasi", "🎛️ Pengaturan", "📊 Riwayat", "👥 Wajah", "🔑 Admin" 
])

# ====================================================
# TAB 1: REGISTRASI (SAMA)
# ====================================================
with tab1:
    c_left, c_center, c_right = st.columns([1, 2, 1])
    with c_center:
        st.header("Pendaftaran Karyawan")
        username = st.text_input("Nama Karyawan Baru")

        if 'reg_data' not in st.session_state: st.session_state['reg_data'] = [] 
        if 'step' not in st.session_state: st.session_state['step'] = 0

        instructions = ["😐 Datar", "😁 Senyum", "↗️ Kanan", "↖️ Kiri", "⬆️ Atas", "⬇️ Bawah", "🤪 Miring Kanan", "🤪 Miring Kiri"]
        current = st.session_state['step']

        if current < 8:
            st.info(f"**Langkah {current + 1}/8:** {instructions[current]}")
            img_file = st.camera_input("Ambil Foto", key=f"cam_{current}")
            
            if img_file:
                bytes_data = img_file.getvalue()
                raw_cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
                cv_img = cv2.flip(raw_cv_img, 1)
                coords = engine.extract_face_coords(cv_img)
                
                if coords:
                    face_crop = cv_img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
                    emb = engine.get_embedding(face_crop)
                    st.session_state['reg_data'].append(emb)
                    st.session_state['step'] += 1
                    st.toast("Tersimpan!", icon="✅")
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.success("✅ Data Lengkap!")
            if st.button("💾 Simpan", type="primary"):
                master_emb = engine.calculate_average_embedding(st.session_state['reg_data'])
                if db.save_user(username, master_emb):
                    st.balloons()
                    st.success(f"{username} terdaftar!")
                    st.session_state['reg_data'] = []
                    st.session_state['step'] = 0
                    st.rerun()
            if st.button("Ulangi"):
                st.session_state['reg_data'] = []
                st.session_state['step'] = 0
                st.rerun()

# ====================================================
# TAB 2: PENGATURAN SISTEM (FINAL LAYOUT FIX)
# ====================================================
with tab2:
    st.header("Konfigurasi Global")
    
    @st.cache_data(ttl=10)
    def load_config(): return config_mgr.get_config()
    current_conf = load_config()

    # 1. INISIALISASI SESSION STATE KOORDINAT
    if 'map_lat' not in st.session_state:
        st.session_state['map_lat'] = float(current_conf.get('office_lat', -7.25))
    if 'map_lon' not in st.session_state:
        st.session_state['map_lon'] = float(current_conf.get('office_lon', 112.75))

    col_loc, col_ai = st.columns(2) # KOLOM UTAMA

    # === BLOK PETA (KOLOM KIRI) ===
    with col_loc:
        st.subheader("📍 Lokasi Kantor")
        st.info("Gunakan **Toolbar Kotak** di kiri peta -> Klik **Ikon Pin** -> Tancapkan di peta.")
        
        # Buat Peta
        m = folium.Map(location=[st.session_state['map_lat'], st.session_state['map_lon']], zoom_start=16)
        folium.Marker(
            [st.session_state['map_lat'], st.session_state['map_lon']], 
            tooltip="Lokasi Terpilih", icon=folium.Icon(color="red", icon="home")
        ).add_to(m)

        # AKTIFKAN DRAWING TOOL
        draw = Draw(
            draw_options={'polyline': False, 'polygon': False, 'circle': False, 'rectangle': False, 'circlemarker': False, 'marker': True},
            edit_options={'edit': False}
        )
        draw.add_to(m)

        # RENDER PETA (HEIGHT DIJAGA TETAP 550)
        output = st_folium(m, height=550, width=700, key="draw_map", returned_objects=["all_drawings"])

        # LOGIKA TANGKAP PIN
        if output and output.get("all_drawings"):
            drawings = output["all_drawings"]
            if len(drawings) > 0:
                last_draw = drawings[-1]
                if last_draw['geometry']['type'] == 'Point':
                    new_lon, new_lat = last_draw['geometry']['coordinates']
                    if (abs(new_lat - st.session_state['map_lat']) > 0.00001) or (abs(new_lon - st.session_state['map_lon']) > 0.00001):
                        st.session_state['map_lat'] = new_lat
                        st.session_state['map_lon'] = new_lon
                        st.toast("Koordinat Baru Tertangkap! 📍", icon="✅")
                        time.sleep(1)
                        st.rerun()

    # === BLOK AI SENSITIVITY (KOLOM KANAN, DI SAMPING PETA) ===
    with col_ai:
        st.subheader("🧠 Sensitivitas AI")
        st.markdown('<p class="dark-blue-text">Threshold Wajah</p>', unsafe_allow_html=True)
        # Slider ditaruh di sini agar posisinya SEJAJAR dengan Peta
        face_thresh = st.slider("", 0.0, 1.0, float(current_conf.get('face_threshold', 0.70)), 0.01, label_visibility="collapsed", key="ai_slider_visual")

        st.markdown('---')
        st.markdown('**Koordinat Terpilih**')
        # Tampilkan nilai koordinat saat ini (sebagai informasi)
        st.info(f"Lat: {st.session_state['map_lat']:.6f} | Lon: {st.session_state['map_lon']:.6f}")
        
    # === BLOK FORM (DI BAWAH KOLOM UTAMA UNTUK SUBMIT) ===
    # Kita buat form di bawah, tapi isinya hanya sisa input yang perlu dikirim
    with st.form("edit_config"):
        
        # Ambil nilai dari Session State (yang diupdate Peta)
        lat = st.session_state['map_lat']
        lon = st.session_state['map_lon']
        
        # Ambil nilai Threshold dari Slider di atas
        # Kita panggil nilai dari Session State secara manual
        # Streamlit menyimpan nilai slider di session_state[key]
        final_face_thresh = st.session_state.get("ai_slider_visual", float(current_conf.get('face_threshold', 0.70)))

        st.subheader("Simpan Konfigurasi Lain")
        
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown('**Radius Absensi (km)**')
            rad = st.number_input("", value=float(current_conf.get('radius_km', 0.5)), step=0.1, label_visibility="collapsed")
        
        with col_f2:
            st.empty() # Placeholder agar Submit Button sejajar

        # Hidden Inputs
        hidden_start = current_conf.get('start_time', "08:00")
        hidden_late = current_conf.get('late_tolerance_time', "09:00")
        hidden_cutoff = current_conf.get('cutoff_time', "10:00")

        if st.form_submit_button("Simpan Konfigurasi", use_container_width=True):
            # Coba simpan dengan semua parameter (termasuk yang disembunyikan)
            try:
                success = config_mgr.save_config(lat, lon, rad, final_face_thresh, hidden_start, hidden_late, hidden_cutoff)
            except TypeError:
                success = config_mgr.save_config(lat, lon, rad, final_face_thresh, 0.0)

            if success:
                load_config.clear()
                st.success("✅ Tersimpan!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Gagal update.")

# ====================================================
# TAB 3: RIWAYAT (SAMA)
# ====================================================
with tab3:
    st.header("📊 Data Log")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()
    
    df_logs = logger.get_logs(limit=100)
    if not df_logs.empty:
        st.dataframe(df_logs, use_container_width=True, hide_index=True)
        csv = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button("📥 CSV", data=csv, file_name="logs.csv", mime='text/csv')
    else:
        st.info("Kosong.")

# ====================================================
# TAB 4: KELOLA WAJAH (SAMA)
# ====================================================
with tab4:
    st.header("👥 Database Wajah")
    users = db.get_all_users()
    
    if users:
        sel_user = st.selectbox("Pilih Karyawan:", users)
        if sel_user:
            vars = db.get_user_variations(sel_user)
            st.write(f"Variasi Wajah: **{len(vars)}**")
            
            if vars:
                df_v = pd.DataFrame(vars)
                st.dataframe(df_v, use_container_width=True)
                
                # Hapus Variasi
                opts = {f"{v['created_at']} ({v['id'][:4]}..)": v['id'] for v in vars}
                del_v = st.selectbox("Hapus variasi:", list(opts.keys()))
                if st.button("Hapus Variasi Ini"):
                    if db.delete_point(opts[del_v]):
                        st.success("Dihapus!"); time.sleep(1); st.rerun()
            
            st.divider()
            if st.button(f"🔥 Hapus User {sel_user} Permanen"):
                if db.delete_user(sel_user):
                    st.success("User Dihapus."); time.sleep(1); st.rerun()
    else:
        st.warning("Database Kosong.")

# ====================================================
# TAB 5: ADMIN (SAMA)
# ====================================================
with tab5:
    st.header("🔑 Akun Admin")
    df_adm = pd.DataFrame(auth.get_all_admins())
    st.dataframe(df_adm, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tambah")
        u = st.text_input("User Baru")
        p = st.text_input("Pass Baru", type="password")
        if st.button("Tambah Admin"):
            if auth.add_admin(u, p)[0]: st.success("Ok!"); time.sleep(1); st.rerun()
            else: st.error("Gagal")
            
    with c2:
        st.subheader("Hapus")
        valid = [x['username'] for x in auth.get_all_admins() if x['username'] != st.session_state['admin_name']]
        if valid:
            d = st.selectbox("Hapus Admin:", valid)
            if st.button("Hapus"):
                if auth.delete_admin(d): st.success("Ok!"); time.sleep(1); st.rerun()
