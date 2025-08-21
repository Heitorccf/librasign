# -*- coding: utf-8 -*-
"""
Script para Predição em Tempo Real com Arquitetura Multithread.

Este módulo implementa a aplicação final com otimizações de performance.
A captura de vídeo e a inferência do modelo rodam em threads separadas,
comunicando-se através de uma fila (Queue). Isso garante que a interface
do usuário permaneça fluida e responsiva, mesmo com modelos mais pesados.
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
# --- MUDANÇA: Importando threading e queue para o processamento paralelo ---
import threading
from queue import Queue

print("[DEBUG] Script de predição iniciado.")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Etapa 1: Configurações e Carregamento ---

MODEL_PATH = "models/best_model_mobilenet.keras" # Carrega o novo modelo
IMG_SIZE = 224

print(f"[INFO] Carregando modelo de: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("[DEBUG] Modelo carregado com sucesso.")

# Recria o mapeamento de rótulos
labels_path = "data/raw"
labels = sorted(os.listdir(labels_path))
# O LabelBinarizer não é mais necessário aqui, podemos usar a lista de labels diretamente.

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Etapa 2: Configuração da Arquitetura Multithread ---

# A fila é o nosso "canal de comunicação" entre as threads.
# A thread de captura coloca imagens aqui, e a de predição as retira.
prediction_queue = Queue(maxsize=1)
# Variável global para armazenar a última predição.
prediction_label = ""

def prediction_worker():
    """Função que roda na thread de predição (worker)."""
    global prediction_label
    while True:
        # Pega uma imagem da fila. O `get()` bloqueia a thread até que um item esteja disponível.
        hand_img_roi = prediction_queue.get()
        if hand_img_roi is None: # Sinal para parar a thread
            break

        # --- Pré-processamento para o MobileNetV2 ---
        # Converte a ROI para RGB (mesmo que a original seja BGR).
        hand_img_rgb = cv2.cvtColor(hand_img_roi, cv2.COLOR_BGR2RGB)
        hand_img_resized = cv2.resize(hand_img_rgb, (IMG_SIZE, IMG_SIZE))
        # Normaliza e expande as dimensões para o formato do modelo (1, 224, 224, 3)
        hand_img_normalized = hand_img_resized.astype(np.float32) / 255.0
        hand_img_expanded = np.expand_dims(hand_img_normalized, axis=0)
        
        # Realiza a inferência
        prediction = model.predict(hand_img_expanded, verbose=0)
        predicted_index = np.argmax(prediction)
        
        # Atualiza a variável global com o resultado
        prediction_label = labels[predicted_index]

# Inicia a thread de predição. `daemon=True` faz com que ela feche quando o programa principal fechar.
threading.Thread(target=prediction_worker, daemon=True).start()

# --- Etapa 3: Loop Principal (Thread de Captura e UI) ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRO CRÍTICO] Falha ao acessar a webcam.")
    exit()

print("[INFO] Sistema pronto. Pressione 'ESC' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    # Esta thread apenas detecta a mão e coloca na fila. Ela não espera pela predição.
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            margin = 20
            x_min, x_max = int(min(x_coords)) - margin, int(max(x_coords)) + margin
            y_min, y_max = int(min(y_coords)) - margin, int(max(y_coords)) + margin
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)
            
            hand_img_roi = frame[y_min:y_max, x_min:x_max]

            # Se a fila estiver vazia, coloca a nova ROI para ser processada.
            # Isso evita que a thread de predição fique sobrecarregada com imagens antigas.
            if prediction_queue.empty() and hand_img_roi.size > 0:
                prediction_queue.put(hand_img_roi)

    # --- Seção de Exibição da Interface (UI) ---
    # A UI apenas lê a variável global `prediction_label` e a exibe.
    # Ela não se importa com o quão rápido ou lento o modelo é.
    display_text = f"Letra: {prediction_label}" if prediction_label else "Aguardando gesto..."
    bg_color = (0, 128, 0) if prediction_label else (128, 0, 0)

    (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_TRIPLEX, 1.0, 2)
    rect_start = (frame.shape[1] - text_width - 30, 10)
    rect_end = (frame.shape[1] - 10, 20 + text_height)
    text_pos = (frame.shape[1] - text_width - 20, 10 + text_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, rect_start, rect_end, bg_color, cv2.FILLED)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, display_text, text_pos, cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Reconhecimento de Gestos - LIBRAS (Avançado)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# --- Finalização ---
print("[DEBUG] Saindo do loop principal e liberando recursos.")
# Envia um sinal para a thread de predição parar
prediction_queue.put(None)
cap.release()
cv2.destroyAllWindows()
print("[DEBUG] Fim do script.")