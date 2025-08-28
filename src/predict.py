# -*- coding: utf-8 -*-
"""
Sistema de Inferência em Tempo Real para Reconhecimento de Gestos em LIBRAS

Este módulo implementa a aplicação prática do modelo treinado de rede neural,
realizando classificação contínua de gestos capturados através da câmera,
com processamento e normalização dos pontos de referência anatômicos da mão.
"""
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Carregando o modelo treinado e o normalizador de características
print("[INFO] Carregando modelo MLP e normalizador...")
with open("models/librasign_mlp.pkl", 'rb') as f:
    model = pickle.load(f)
with open("models/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Configurando os componentes de detecção e visualização do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("[INFO] Sistema pronto. Pressione 'ESC' para sair.")

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    prediction_label = ""
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extraindo as coordenadas tridimensionais e preparando para inferência
            landmarks = hand_landmarks.landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)
            scaled_coords = scaler.transform(coords)
            
            # Executando a classificação através do modelo treinado
            prediction = model.predict(scaled_coords)
            prediction_label = prediction[0]
    
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