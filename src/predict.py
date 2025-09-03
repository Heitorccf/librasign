# -*- coding: utf-8 -*-
"""
Sistema de Inferência em Tempo Real para Reconhecimento de Gestos em LIBRAS.

Este módulo implementa a aplicação do modelo treinado, realizando classificação
de gestos capturados pela câmera. Inclui:
1. Normalização de landmarks em tempo real para compatibilidade com o modelo.
2. Filtro de suavização temporal para estabilizar as predições.
3. Exibição da confiança do modelo para feedback ao usuário.
"""
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque

# --- 1. FUNÇÃO DE NORMALIZAÇÃO DE LANDMARKS (IDÊNTICA À DO TREINAMENTO) ---
def normalize_landmarks(data):
    """
    Normaliza um único conjunto de 63 coordenadas de landmarks.
    """
    landmarks = data.reshape((21, 3))
    base_point = landmarks[0].copy()
    relative_landmarks = landmarks - base_point
    
    scale_dist = np.linalg.norm(relative_landmarks[9])
    if scale_dist < 1e-6:
        scale_dist = 1.0

    scaled_landmarks = relative_landmarks / scale_dist
    return scaled_landmarks.flatten()

# --- 2. CARREGAMENTO DOS ARTEFATOS DO MODELO ---
print("[INFO] Carregando modelo MLP, normalizador e classes...")
with open("models/librasign_mlp.pkl", 'rb') as f:
    model = pickle.load(f)
with open("models/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

class_names = np.load("models/classes.npy")

# --- 3. CONFIGURAÇÕES DO SISTEMA DE PREDIÇÃO ---
# Configurações do MediaPipe
mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Configurações do Filtro de Suavização e Confiança
HISTORY_SIZE = 10  # Usar os últimos 10 frames para suavizar
CONFIDENCE_THRESHOLD = 0.75 # Limiar de 75% de confiança para exibir a letra
predictions_history = deque(maxlen=HISTORY_SIZE)

cap = cv2.VideoCapture(0)
print("[INFO] Sistema pronto. Pressione 'ESC' para sair.")

# --- 4. LOOP DE INFERÊNCIA EM TEMPO REAL ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    current_label = ""
    confidence = 0.0
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extrair e normalizar landmarks
            coords_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            coords_normalized = normalize_landmarks(coords_raw).reshape(1, -1)
            
            # Escalar com o StandardScaler
            scaled_coords = scaler.transform(coords_normalized)
            
            # Fazer a predição de probabilidade
            prediction_proba = model.predict_proba(scaled_coords)
            prediction_index = np.argmax(prediction_proba)
            confidence = prediction_proba[0][prediction_index]
            
            # Adicionar ao histórico apenas se a confiança for alta
            if confidence >= CONFIDENCE_THRESHOLD:
                current_label = class_names[prediction_index]
                predictions_history.append(current_label)
            else:
                predictions_history.append(None) # Adiciona None se a confiança for baixa

    # Lógica de suavização (votação majoritária)
    smoothed_label = ""
    if predictions_history:
        # Filtra os Nones e encontra o mais comum
        valid_preds = [p for p in predictions_history if p is not None]
        if valid_preds:
            smoothed_label = max(set(valid_preds), key=valid_preds.count)

    # Construindo a interface visual de feedback
    if smoothed_label and confidence >= CONFIDENCE_THRESHOLD:
        display_text = f"Letra: {smoothed_label} ({confidence:.0%})"
    else:
        display_text = "Aguardando gesto..."

    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imshow("Predição com Landmarks - LIBRASIGN", frame)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Aplicação encerrada.")