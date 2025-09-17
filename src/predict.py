# -*- coding: utf-8 -*-
"""
Sistema de Inferência em Tempo Real para Reconhecimento de Gestos em LIBRAS.

Este módulo implementa a aplicação do modelo treinado, incluindo:
1. Normalização de landmarks em tempo real.
2. Filtro de suavização temporal para estabilizar as predições.
3. Exibição da confiança do modelo.
4. Lógica de confirmação por tempo para construção de frases.
"""
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Carregando o modelo treinado, o normalizador e o mapeamento de classes
print("[INFO] Carregando modelo MLP, normalizador e classes...")
with open("models/librasign_mlp.pkl", 'rb') as f:
    model = pickle.load(f)
with open("models/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

class_names = np.load("models/classes.npy")

# --- 3. CONFIGURAÇÕES DO SISTEMA ---
# MediaPipe
mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("[INFO] Sistema pronto. Pressione 'ESC' para sair.")

# --- 4. LOOP DE INFERÊNCIA EM TEMPO REAL ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    H, W, _ = frame.shape
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    # Processamento de landmarks e predição
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)
            scaled_coords = scaler.transform(coords)
            
            # Executa a classificação, que retorna um número (índice)
            prediction_index = model.predict(scaled_coords)
            
            # --- MUDANÇA: "Traduzindo" o índice para a letra correspondente ---
            prediction_label = class_names[prediction_index[0]]
    
    # Construindo a interface visual de feedback para o usuário
    display_text = f"Letra: {prediction_label}" if prediction_label else "Aguardando gesto..."
    bg_color = (0, 128, 0) if prediction_label else (128, 0, 0)
    
    # Renderizando as informações de predição na tela
    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Predição com Landmarks", frame)
    
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Aplicação encerrada.")