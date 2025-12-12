import streamlit as st
from supabase import create_client, Client

class ConfigManager:
    def __init__(self):
        try:
            url = st.secrets["supabase"]["URL"]
            key = st.secrets["supabase"]["KEY"]
            self.supabase: Client = create_client(url, key)
        except Exception as e:
            st.error(f"Error Config Init: {e}")

    def get_config(self):
        """Ambil semua config dari tabel 'config'"""
        try:
            # ... (kode lama) ...
            response = self.supabase.table("config").select("*").execute()
            data = response.data
            
            config_dict = {item['key']: item['value'] for item in data}
            return config_dict
            
        except Exception:
            # --- DEFAULT BARU ---
            return {
                "office_lat": -7.2575,
                "office_lon": 112.7521,
                "radius_km": 0.5,
                "face_threshold": 0.65,
                "liveness_threshold": 60.0,
                "max_time_in": "07:00",         # <--- BARU: Batas Absen Masuk
                "min_duration_hours": "8"       # <--- BARU: Durasi Kerja Minimum
            }

    # Ubah signature fungsi untuk menerima dua parameter baru
    def save_config(self, lat, lon, radius, face_thresh, liveness_thresh, max_in_time, min_out_hours):
        """Update config satu per satu (Upsert)"""
        try:
            updates = [
                {"key": "office_lat", "value": str(lat)},
                {"key": "office_lon", "value": str(lon)},
                {"key": "radius_km", "value": str(radius)},
                {"key": "face_threshold", "value": str(face_thresh)}, 
                {"key": "liveness_threshold", "value": str(liveness_thresh)},
                # --- TAMBAHAN BARU ---
                {"key": "max_time_in", "value": max_in_time},
                {"key": "min_duration_hours", "value": str(min_out_hours)}
                # ---------------------
            ]
            self.supabase.table("config").upsert(updates).execute()
            return True
        except Exception as e:
            st.error(f"Gagal update config: {e}")
            return False