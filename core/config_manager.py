import streamlit as st
from supabase import create_client, Client

class ConfigManager:
    def __init__(self):
        try:
            # Menggunakan st.secrets untuk URL dan KEY Supabase
            url = st.secrets["supabase"]["URL"]
            key = st.secrets["supabase"]["KEY"]
            self.supabase: Client = create_client(url, key)
        except Exception as e:
            # Tampilkan error jika inisialisasi gagal (misalnya, masalah secrets)
            st.error(f"Error Config Init: {e}")

    def get_config(self):
        """Ambil semua config dari tabel 'config' dan berikan nilai default."""
        
        # Nilai default dasar yang akan digunakan jika Supabase gagal atau nilai tidak ada
        DEFAULT_CONFIG = {
            "office_lat": -7.2575,
            "office_lon": 112.7521,
            "radius_km": 0.5,
            "face_threshold": 0.70,
            "liveness_threshold": 0.0,
            
            # Parameter Jam Kerja
            "start_time": "08:00",          # Jam Masuk Standar
            "late_tolerance_time": "09:00", # Batas Terlambat (untuk perhitungan keterlambatan)
            "cutoff_time": "10:00",         # Batas Akhir Absensi Masuk (tidak bisa absen setelah ini)
            
            # Parameter Batas Waktu/Durasi Tambahan (dari fitur-batas-waktu-v2)
            # Karena ini nilai baru, kita tetapkan default dari sini.
            "max_time_in": "10:00",         # Batas Absen Masuk (sama dengan cutoff_time agar konsisten)
            "min_duration_hours": "8"       # Durasi Kerja Minimum (dalam jam string)
        }

        try:
            # Panggil data dari Supabase
            response = self.supabase.table("config").select("*").execute()
            data = response.data
            
            # Ubah list of dicts menjadi dict untuk akses mudah
            config_dict = {item['key']: item['value'] for item in data}
            
            # Gabungkan dengan nilai default: ambil dari DB jika ada, jika tidak, ambil dari DEFAULT_CONFIG
            
            # Terapkan konversi tipe data yang sesuai
            return {
                "office_lat": float(config_dict.get("office_lat", DEFAULT_CONFIG["office_lat"])),
                "office_lon": float(config_dict.get("office_lon", DEFAULT_CONFIG["office_lon"])),
                "radius_km": float(config_dict.get("radius_km", DEFAULT_CONFIG["radius_km"])),
                "face_threshold": float(config_dict.get("face_threshold", DEFAULT_CONFIG["face_threshold"])),
                "liveness_threshold": float(config_dict.get("liveness_threshold", DEFAULT_CONFIG["liveness_threshold"])),
                
                "start_time": config_dict.get("start_time", DEFAULT_CONFIG["start_time"]),
                "late_tolerance_time": config_dict.get("late_tolerance_time", DEFAULT_CONFIG["late_tolerance_time"]),
                "cutoff_time": config_dict.get("cutoff_time", DEFAULT_CONFIG["cutoff_time"]),
                
                "max_time_in": config_dict.get("max_time_in", DEFAULT_CONFIG["max_time_in"]),
                "min_duration_hours": config_dict.get("min_duration_hours", DEFAULT_CONFIG["min_duration_hours"])
            }
            
        except Exception as e:
            # Jika Supabase gagal atau ada error lain, berikan semua default
            st.error(f"Gagal mengambil config dari Supabase. Menggunakan nilai default: {e}")
            return DEFAULT_CONFIG

    # Menggabungkan semua 10 parameter (termasuk liveness_thresh yang diset 0.0 di dalam)
    def save_config(self, lat, lon, radius, face_thresh, start_time, late_tolerance_time, cutoff_time, max_in_time, min_out_hours):
        """Update semua 10 parameter config utama (liveness_threshold dihardcode ke 0.0)."""
        try:
            # Hardcode liveness_threshold ke 0.0 sesuai kebutuhan penghapusan fitur
            liveness_thresh = 0.0 
            
            updates = [
                {"key": "office_lat", "value": str(lat)},
                {"key": "office_lon", "value": str(lon)},
                {"key": "radius_km", "value": str(radius)},
                {"key": "face_threshold", "value": str(face_thresh)}, 
                {"key": "liveness_threshold", "value": str(liveness_thresh)}, # Dihardcode 0.0
                
                # Parameter Jam Kerja (dari Main)
                {"key": "start_time", "value": str(start_time)}, 
                {"key": "late_tolerance_time", "value": str(late_tolerance_time)},
                {"key": "cutoff_time", "value": str(cutoff_time)},
                
                # Parameter Batas Waktu/Durasi Tambahan (dari fitur-batas-waktu-v2)
                {"key": "max_time_in", "value": str(max_in_time)},
                {"key": "min_duration_hours", "value": str(min_out_hours)}
            ]
            
            # Supabase upsert (Insert or Update)
            self.supabase.table("config").upsert(updates).execute()
            return True
        except Exception as e:
            st.error(f"Gagal update config: {e}")
            return False