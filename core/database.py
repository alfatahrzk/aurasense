# core/database.py
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

class VectorDB:
    def __init__(self):
        self.client = None # Inisialisasi awal None agar tidak AttributeError
        
        try:
            # Cek apakah secrets ada
            if "QDRANT_URL" not in st.secrets:
                st.error("❌ QDRANT_URL belum disetting di Secrets Streamlit Cloud!")
                return

            self.client = QdrantClient(
                url=st.secrets["QDRANT_URL"],
                api_key=st.secrets["QDRANT_API_KEY"]
            )
            self.collection_name = "absensi"
            self._init_collection()
            
        except Exception as e:
            st.error(f"🔥 Gagal konek ke Qdrant: {e}")

    def _init_collection(self):
        if self.client: # Cek client ada
            try:
                self.client.get_collection(self.collection_name)
            except:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )

    def save_user(self, username, embedding):
        if not self.client: return False # Cegah crash
        
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={"username": username, "role": "user"}
                )
            ]
        )
        return True

    def search_user(self, embedding, threshold=0.5):
        if self.client is None:
            st.error("⚠️ Database Wajah Offline.")
            return None, 0.0
        
        # --- DIAGNOSTIK: CEK ISI PERUT QDRANT ---
        # Kita akan melihat daftar method yang tersedia
        available_methods = dir(self.client)
        
        # Cek apakah ada kata 'search' di dalam daftar tersebut
        has_search = 'search' in available_methods
        
        if not has_search:
            st.error("SOS! Method .search() HILANG dari QdrantClient!")
            st.write("Daftar method yang tersedia (Cek apakah ada yang mirip):")
            st.code(available_methods) # Tampilkan semua ke layar
            st.stop()
        # ----------------------------------------

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                limit=1,
                score_threshold=threshold
            )
            if results:
                return results[0].payload['username'], results[0].score
            return None, 0.0
        except Exception as e:
            st.error(f"Error search: {e}")
            return None, 0.0