import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
import streamlit as st
from facenet_pytorch import MTCNN

# --- ARSITEKTUR MODEL (TETAP) ---
class _IndonesianFaceModel(nn.Module):
    def __init__(self, num_classes=68):
        super(_IndonesianFaceModel, self).__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        input_dim = 2048
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc_embedding = nn.Linear(input_dim, 512)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn_input(x)
        x = self.dropout(x)
        return self.fc_embedding(x)

# --- ENGINE UTAMA ---
class FaceEngine:
    def __init__(self, model_path='models/indonesian_arcface_r50.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = _IndonesianFaceModel(num_classes=68)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            self.model.to(self.device)
            self.model.eval()
            if self.device.type == 'cpu':
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
        except Exception as e:
            print(f"Model Load Error: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.detector = MTCNN(keep_all=True, device=self.device, min_face_size=40, thresholds=[0.6, 0.7, 0.7])

    def extract_face_coords(self, image_cv2):
        if image_cv2 is None: return None
        
        scale_factor = 0.5
        small_img = cv2.resize(image_cv2, (0, 0), fx=scale_factor, fy=scale_factor)
        
        height, width, _ = small_img.shape
        img_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        try:
            boxes, probs = self.detector.detect(img_pil)
            if boxes is None or len(boxes) == 0: return None
            
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            x1, y1, x2, y2 = [int(b) for b in box]
            
            real_x = int(max(0, x1) / scale_factor)
            real_y = int(max(0, y1) / scale_factor)
            real_w = int(min(image_cv2.shape[1] - max(0, x1), x2 - x1) / scale_factor)
            real_h = int(min(image_cv2.shape[0] - max(0, y1), y2 - y1) / scale_factor)
            
            if real_w < 20 or real_h < 20: return None
            return (real_x, real_y, real_w, real_h)
        except Exception:
            return None

    def get_embedding(self, face_crop):
        if face_crop is None or face_crop.size == 0: return np.zeros(512)
        img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding.cpu().numpy()[0]
    
    def calculate_average_embedding(self, embeddings_list):
        if not embeddings_list: return None
        stack = np.stack(embeddings_list)
        mean_emb = np.mean(stack, axis=0)
        return mean_emb / np.linalg.norm(mean_emb)

    # --- FITUR ANTI SPOOFING V8 (FULL FRAME CONTEXT) ---
    def check_liveness(self, face_crop, full_image):
        """
        Analisis Gabungan: Crop Wajah + Full Frame Context
        """
        if face_crop is None or full_image is None: return False, 0.0
            
        # 1. Analisis Crop Wajah (Laplacian) - Fokus ke detail kulit
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # 2. Analisis Full Frame (Laplacian) - Fokus ke ketajaman global
        # Layar laptop biasanya tajam merata (nilai tinggi di seluruh gambar)
        gray_full = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        # Resize biar cepet hitungnya
        small_full = cv2.resize(gray_full, (0, 0), fx=0.5, fy=0.5) 
        global_score = cv2.Laplacian(small_full, cv2.CV_64F).var()
        
        # 3. Analisis Frekuensi Full Frame (Moiré Pattern Global)
        # Layar memancarkan pola grid di seluruh permukaan, bukan cuma di wajah
        f = np.fft.fft2(small_full)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)
        global_fourier = np.mean(magnitude_spectrum)
        
        # --- LOGIKA PENILAIAN BARU ---
        
        final_score = face_score # Kita mulai dari skor wajah
        
        # PENALTI 1: GLOBAL SHARPNESS (Layar vs Asli)
        # Foto Asli: Wajah tajam, Background agak blur (Bokeh).
        # Foto Layar: Semuanya tajam (Datar).
        # Jika Global Score sangat tinggi (> 100), kemungkinan layar.
        if global_score > 100:
            final_score -= 30 
            
        # PENALTI 2: EXTREME FACE SHARPNESS (Layar Retina)
        # Ini menampung kasus skor 300+ yang Anda alami
        if face_score > 150:
            # Semakin tinggi di atas 150, semakin palsu.
            # Kita diskon habis-habisan.
            final_score = final_score * 0.3 
            
        # PENALTI 3: GLOBAL MOIRE
        # Pola digital di seluruh frame
        if global_fourier > 160:
            final_score -= 40
            
        # PENALTI 4: FACE RATIO (Kepala Melayang)
        # Jika wajah terlalu memenuhi frame (Zoom layar), kurangi nilai
        h_full, w_full = full_image.shape[:2]
        h_face, w_face = face_crop.shape[:2]
        face_area_ratio = (h_face * w_face) / (h_full * w_full)
        
        # Jika wajah > 60% dari total foto, curiga (biasanya orang selfie ada bahu/background)
        if face_area_ratio > 0.6:
            final_score -= 20

        # --- NORMALISASI ---
        final_score = max(0.0, min(100.0, final_score))
        
        # Threshold Lulus
        is_real = final_score > 50.0
        
        return is_real, final_score