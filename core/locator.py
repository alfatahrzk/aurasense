# core/locator.py
import streamlit as st
from streamlit_js_eval import get_geolocation
import requests
# Ganti import: Dari Nominatim ke ArcGIS
from geopy.geocoders import ArcGIS 

class LocationService:
    def __init__(self):
        # Menggunakan ArcGIS karena lebih stabil dan tidak gampang Connection Refused
        self.geolocator = ArcGIS()

    def get_coordinates(self):
        """
        Mencoba mendapatkan lokasi: GPS Browser -> IP Address
        """
        lat, lon, source = None, None, None

        # 1. GPS BROWSER
        try:
            loc_data = get_geolocation(component_key='get_gps_loc')
            
            if loc_data and 'coords' in loc_data:
                lat = loc_data['coords']['latitude']
                lon = loc_data['coords']['longitude']
                source = "GPS (Akurasi Tinggi)"
                return lat, lon, source
        except:
            pass # Lanjut ke fallback jika JS gagal

        # 2. IP ADDRESS (FALLBACK)
        try:
            # Ganti timeout jadi lebih pendek biar gak lama nunggu
            response = requests.get('http://ip-api.com/json/', timeout=2)
            if response.status_code == 200:
                data = response.json()
                lat = data['lat']
                lon = data['lon']
                source = "IP Address (Estimasi)"
                return lat, lon, source
        except Exception as e:
            print(f"Gagal IP Location: {e}")

        return None, None, None

    def get_address(self, lat, lon):
        """
        Mengubah Lat/Lon menjadi Alamat Lengkap (Reverse Geocoding)
        """
        try:
            # ArcGIS tidak butuh parameter language='id', dia otomatis deteksi
            location = self.geolocator.reverse((lat, lon), timeout=10)
            
            if location:
                return location.address
            else:
                return "Alamat tidak ditemukan"
                
        except Exception as e:
            # JIKA MASIH ERROR, KITA KEMBALIKAN KOORDINAT SAJA
            # Agar aplikasi TIDAK CRASH hanya gara-gara alamat gagal dimuat
            return f"Koordinat: {lat:.5f}, {lon:.5f}"