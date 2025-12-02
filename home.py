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
            background-color: #004080; /* Biru sedikit lebih cerah dari header */
            padding: 10px 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .navbar a {
            color: #ffffff !important;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="header"><h1>🏢 AuraSense Presence</h1></div>', unsafe_allow_html=True)

# Navigation links dengan background biru (navbar)
st.markdown('<div class="navbar">', unsafe_allow_html=True)
nav_col1, nav_col2 = st.columns([1, 1])
with nav_col1:
    st.page_link("home.py", label="🏠 Home")
with nav_col2:
    st.page_link("pages/Absensi.py", label="📸 Absen")
st.markdown('</div>', unsafe_allow_html=True)

# Main Content
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3652/3652191.png", width=150)
    with col2:
        st.markdown("""
        <div class='content'>
            <h3 style='color: #003366; margin-top: 0;'>Selamat Datang di AuraSense</h3>
            <p style='color: #003366;'>Sistem ini menggunakan teknologi <strong>Face Recognition berbasis AI (ResNet50)</strong> 
            with penyimpanan <strong>Vector Database (Qdrant)</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='content' style='margin-top: 20px;'>
        <h3 style='color: #003366;'>Menu Utama</h3>
        <p style='color: #003366;'>Silakan pilih menu di sidebar (sebelah kiri):</p>
        <ul style='color: #003366;'>
            <li><strong>Registrasi Wajah:</strong> (Khusus Admin) Untuk mendaftarkan karyawan baru dengan 8 pose.</li>
            <li><strong>Absensi User:</strong> (Akan dibuat selanjutnya) Untuk melakukan presensi harian.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Info box at the bottom
st.info("💡 Pastikan Anda memiliki akses internet stabil untuk terhubung ke Cloud Database.")
