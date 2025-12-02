import streamlit as st
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone # <--- Import tambahan

class AttendanceLogger:
    def __init__(self):
        try:
            url = st.secrets["supabase"]["URL"]
            key = st.secrets["supabase"]["KEY"]
            self.supabase: Client = create_client(url, key)
        except Exception as e:
            st.error(f"Gagal konek Supabase: {e}")

    def log_attendance(self, name, status, location_dist, address, lat, lon, similarity, liveness, validation_status="Berhasil"):
        """
        Mencatat log ke Supabase dengan Waktu WIB (UTC+7)
        """
        try:
            # --- PERBAIKAN TIMEZONE (WIB) ---
            # Definisikan Zona Waktu UTC+7
            WIB = timezone(timedelta(hours=7))
            
            # Ambil waktu sekarang dengan zona WIB
            now_wib = datetime.now(WIB)
            
            # Format ke string
            now_str = now_wib.strftime("%Y-%m-%d %H:%M:%S")
            # -------------------------------
            
            data = {
                "nama": name,
                "status": status,
                "waktu_absen": now_str,
                "jarak": f"{location_dist:.4f}",
                "alamat": address,
                "verifikasi": "Wajah (Qdrant)",
                "koordinat": f"{lat}, {lon}",
                "skor_kemiripan": float(similarity),
                "skor_liveness": float(liveness),
                "status_validasi": validation_status 
            }
            
            self.supabase.table("logs").insert(data).execute()
            return True
            
        except Exception as e:
            st.error(f"Gagal simpan log: {e}")
            return False

    # --- FUNGSI GET LOGS (TETAP SAMA) ---
    def get_logs(self, limit=100):
        try:
            response = self.supabase.table("logs")\
                .select("*")\
                .order("id", desc=True)\
                .limit(limit)\
                .execute()
            
            data = response.data
            
            if data:
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Gagal mengambil log: {e}")
            return pd.DataFrame()