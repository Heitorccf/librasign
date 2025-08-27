# -*- coding: utf-8 -*-
"""
Predição em Tempo Real com Modelo MLP Baseado em Landmarks.
"""
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# --- Carregamento do Modelo e Scaler ---
print("[INFO] Carregando modelo MLP e normalizador...")
with open("models/librasign_mlp.pkl", 'rb') as f:
    model = pickle.load(f)
with open("models/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# --- Inicialização ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

print("[INFO] Sistema pronto. Pressione 'ESC' para sair.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    prediction_label = ""
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extrai, normaliza e prepara os landmarks
            landmarks = hand_landmarks.landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)
            scaled_coords = scaler.transform(coords)
            
            # Faz a predição
            prediction = model.predict(scaled_coords)
            prediction_label = prediction[0]
            
    # --- UI (Pode usar a versão melhorada que já tínhamos) ---
    display_text = f"Letra: {prediction_label}" if prediction_label else "Aguardando gesto..."
    bg_color = (0, 128, 0) if prediction_label else (128, 0, 0)
    # ... (resto do código da UI que você preferir) ...
    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Predição com Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()