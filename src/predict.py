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
import time
from collections import deque

# --- 1. FUNÇÃO DE NORMALIZAÇÃO DE LANDMARKS (IDÊNTICA À DO TREINAMENTO) ---
def normalize_landmarks(data):
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

# --- 3. CONFIGURAÇÕES DO SISTEMA ---
# MediaPipe
mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Predição e Suavização
HISTORY_SIZE = 10
CONFIDENCE_THRESHOLD = 0.75
predictions_history = deque(maxlen=HISTORY_SIZE)

# Lógica de Construção de Frase
sentence = []
CONFIRMATION_TIME = 2.0  # 2 segundos para confirmar uma letra
stable_letter_start_time = None
current_stable_letter = ""
letter_confirmed = False

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
            
            coords_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            coords_normalized = normalize_landmarks(coords_raw).reshape(1, -1)
            scaled_coords = scaler.transform(coords_normalized)
            
            prediction_proba = model.predict_proba(scaled_coords)
            prediction_index = np.argmax(prediction_proba)
            confidence = prediction_proba[0][prediction_index]
            
            if confidence >= CONFIDENCE_THRESHOLD:
                current_label = class_names[prediction_index]
                predictions_history.append(current_label)
            else:
                predictions_history.append(None)
    else:
        # Se nenhuma mão for detectada, limpa o histórico para resetar a suavização
        predictions_history.clear()

    # Lógica de suavização
    smoothed_label = ""
    if predictions_history:
        valid_preds = [p for p in predictions_history if p is not None]
        if valid_preds:
            smoothed_label = max(set(valid_preds), key=valid_preds.count)

    # --- 5. LÓGICA DE CONFIRMAÇÃO E CONSTRUÇÃO DE FRASE ---
    if smoothed_label and smoothed_label != current_stable_letter:
        # Uma nova letra estável foi detectada, iniciar cronômetro
        current_stable_letter = smoothed_label
        stable_letter_start_time = time.time()
        letter_confirmed = False
    elif smoothed_label and smoothed_label == current_stable_letter:
        # A mesma letra continua estável, verificar tempo
        if not letter_confirmed and (time.time() - stable_letter_start_time) >= CONFIRMATION_TIME:
            sentence.append(current_stable_letter)
            letter_confirmed = True # Marca como confirmada para não adicionar de novo
    elif not smoothed_label:
        # Nenhuma letra estável, resetar
        current_stable_letter = ""
        stable_letter_start_time = None
        letter_confirmed = False

    # --- 6. RENDERIZAÇÃO E INTERFACE ---
    # Feedback de predição atual
    if current_stable_letter:
        display_text = f"Gesto: {current_stable_letter}"
        if letter_confirmed:
            display_text += " (Confirmado!)"
        elif stable_letter_start_time:
            # Barra de progresso para confirmação
            progress = (time.time() - stable_letter_start_time) / CONFIRMATION_TIME
            cv2.rectangle(frame, (10, 70), (10 + int(progress * 200), 90), (0, 255, 0), -1)
    else:
        display_text = "Aguardando gesto..."
    
    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    # Exibição da frase formada
    cv2.rectangle(frame, (0, H - 60), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, " ".join(sentence), (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow("Librasign - Tradutor de LIBRAS", frame)
    
    # --- 7. CONTROLES DO TECLADO ---
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC para sair
        break
    if key == 8: # Backspace para apagar a última letra
        if sentence:
            sentence.pop()
    if key == ord('c'): # Tecla 'c' para limpar a frase
        sentence.clear()

cap.release()
cv2.destroyAllWindows()
print("[INFO] Aplicação encerrada.")