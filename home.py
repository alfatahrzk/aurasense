# Home.py
import streamlit as st

# Set page config with custom theme
st.set_page_config(
    page_title="AuraSense",
    layout="centered",
    page_icon="🏢"
)

# Custom CSS for styling
st.markdown("""
    <style>
        *, 
        *::before, 
        *::after {
            box-sizing: border-box;
        }
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
        }
        .navbar {
            background-color: #004080;
            padding: 10px 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0 20px 0;
        }
        .navbar a {
            color: #ffffff !important;
            font-weight: 600;
            text-decoration: none;
            margin-right: 20px;
        }
        .navbar a:last-child {
            margin-right: 0;
        }
        .icon-container {
            background-color: #004080;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .text-primary {
            color: #003366;
        }
        .content-margin {
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Header and Navigation Section
st.markdown("""
<div class="header">
    <h1>🏢 AuraSense Presence</h1>
    <nav class="navbar">
        <a href="home.py">🏠 Home</a>
        <a href="pages/Absensi.py">📸 Absen</a>
    </nav>
</div>
""", unsafe_allow_html=True)

# Main Content
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="icon-container"><img src="https://cdn-icons-png.flaticon.com/512/3652/3652191.png" width="150"></div>', unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='content'>
            <h3 class='text-primary' style='margin-top: 0;'>Selamat Datang di AuraSense</h3>
            <p class='text-primary'>Sistem ini menggunakan teknologi <strong>Face Recognition berbasis AI (ResNet50)</strong> 
            with penyimpanan <strong>Vector Database (Qdrant)</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='content content-margin'>
        <h3 class='text-primary'>Menu Utama</h3>
        <p class='text-primary'>Silakan pilih menu di sidebar (sebelah kiri):</p>
        <ul class='text-primary'>
            <li><strong>Registrasi Wajah:</strong> (Khusus Admin) Untuk mendaftarkan karyawan baru dengan 8 pose.</li>
            <li><strong>Absensi User:</strong> (Akan dibuat selanjutnya) Untuk melakukan presensi harian.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Info box at the bottom
st.info("💡 Pastikan Anda memiliki akses internet stabil untuk terhubung ke Cloud Database.")
