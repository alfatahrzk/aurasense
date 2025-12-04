import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
import streamlit as st
from facenet_pytorch import MTCNN

# --- ARSITEKTUR MODEL RESNET (TETAP SAMA) ---
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
    def __init__(self, model_path='models/model-absensi.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = _IndonesianFaceModel(num_classes=68)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            # Optimasi CPU
            if self.device.type == 'cpu':
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
        except Exception as e:
            print(f"Model Load Error: {e}")

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # MTCNN Detektor
        self.detector = MTCNN(keep_all=True, device=self.device, min_face_size=40, thresholds=[0.6, 0.7, 0.7])

    def extract_face_coords(self, image_cv2):
        if image_cv2 is None: return None
        
        # Resize dulu biar cepat deteksinya
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
            
            # Upscale koordinat
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

    # --- FITUR BARU: ANTI SPOOFING V4 (MULTI-LAYER CHECK) ---
    # --- FITUR ANTI SPOOFING V6 (TUNED BY DATA) ---
    def check_liveness(self, face_crop):
        if face_crop is None or face_crop.size == 0: return False, 0.0
            
        # 1. Analisis Tekstur (Laplacian)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Analisis Warna (YCrCb) - Tetap kita pakai
        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        mean_cr = np.mean(cr)
        mean_cb = np.mean(cb)
        
        skin_score = 100
        # Rentang kulit umum
        if not (133 <= mean_cr <= 173) or not (77 <= mean_cb <= 127):
            skin_score = 0 
        
        # 3. Analisis Frekuensi (Fourier) - Tetap kita pakai
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)
        fourier_score = np.mean(magnitude_spectrum)
        
        # --- LOGIKA SCORING BARU (BERDASARKAN LOGS) ---
        final_score = laplacian_score
        
        # PENALTI 1: TERLALU TAJAM (LAYAR LAPTOP)
        # Berdasarkan log: Wajah asli ~60-120. Layar laptop > 150.
        # Kita set batas aman di 160.
        if laplacian_score > 160: 
            # Semakin tinggi di atas 160, semakin besar diskonnya
            # Contoh: Skor 300 -> Diskon 70% -> Jadi 90
            final_score *= 0.4 
            
        # PENALTI 2: MOIRE PATTERN (Grid Layar)
        if fourier_score > 150: 
            final_score -= 40 
            
        # PENALTI 3: WARNA ANEH
        if skin_score == 0: 
            final_score -= 50 
            
        # --- NORMALISASI KE 0-100 ---
        final_score = max(0.0, min(100.0, final_score))
        
        # Ambang batas kelulusan
        # Kita naikkan sedikit threshold default-nya agar lebih aman
        is_real = final_score > 55.0 
        
        return is_real, final_score