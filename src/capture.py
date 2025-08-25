# -*- coding: utf-8 -*-
"""
Ferramenta Avançada para Captura de Dados Visuais.

Este módulo foi reestruturado para facilitar a coleta de um dataset em larga
escala. Implementa um limite de captura por classe, pré-processamento de
imagem em tempo real (equalização de histograma) e um fluxo de trabalho
interativo que permite ao usuário iniciar, pausar e alternar entre a
captura de diferentes gestos sem reiniciar a aplicação.
"""

import cv2
import mediapipe as mp
import os
from datetime import datetime

# --- Configurações ---
DATA_DIR = "data/raw"
IMG_SIZE = 224
CAPTURE_LIMIT = 1000 # Limite de 1000 fotos por gesto

# --- Inicialização de Componentes ---
mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                 min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# --- Variáveis de Controle de Estado ---
capturing = False     # Controla se a captura está ativa ou pausada
current_label = None  # Letra ou classe atual sendo capturada
img_count = 0         # Contador de imagens para a classe atual

print("-" * 50)
print("Ferramenta de Captura de Dataset - LIBRAS")
print("-" * 50)
print("INSTRUÇÕES:")
print(" > Pressione uma tecla (A-Z) para INICIAR a captura para essa letra.")
print(" > Pressione '0' para INICIAR a captura da classe 'Nenhum'.")
print(" > Pressione 'ESPAÇO' para PAUSAR ou RETOMAR a captura.")
print(" > Pressione 'ESC' para FINALIZAR o programa.")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Não foi possível acessar a câmera.")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # Lógica de processamento e salvamento da imagem
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if capturing and current_label and img_count < CAPTURE_LIMIT:
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]

                margin = 20
                x_min, x_max = int(min(x_coords)) - margin, int(max(x_coords)) + margin
                y_min, y_max = int(min(y_coords)) - margin, int(max(y_coords)) + margin
                x_min, y_min = max(x_min, 0), max(y_min, 0)
                x_max, y_max = min(x_max, w), min(y_max, h)

                hand_img_roi = frame[y_min:y_max, x_min:x_max]

                if hand_img_roi.size > 0:
                    gray_hand = cv2.cvtColor(hand_img_roi, cv2.COLOR_BGR2GRAY)
                    equalized_hand = cv2.equalizeHist(gray_hand)
                    resized_hand = cv2.resize(equalized_hand, (IMG_SIZE, IMG_SIZE))

                    save_dir = os.path.join(DATA_DIR, current_label.upper())
                    os.makedirs(save_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    filename = f"{current_label.upper()}_{timestamp}.jpg"
                    cv2.imwrite(os.path.join(save_dir, filename), resized_hand)
                    img_count += 1

    # --- Lógica de Exibição de Status na Tela ---
    status_text = ""
    status_color = (0, 0, 0)

    if current_label:
        progress_text = f"Classe: {current_label.upper()} ({img_count}/{CAPTURE_LIMIT})"
        if capturing:
            if img_count < CAPTURE_LIMIT:
                status_text = "GRAVANDO... (Pressione ESPAÇO para pausar)"
                status_color = (0, 255, 0) # Verde
            else:
                status_text = "LIMITE ATINGIDO! Escolha outra letra."
                status_color = (0, 0, 255) # Vermelho
                capturing = False
        else:
             if img_count < CAPTURE_LIMIT:
                status_text = "PAUSADO (Pressione ESPAÇO para retomar)"
                status_color = (0, 255, 255) # Amarelo
             else:
                status_text = "LIMITE ATINGIDO! Escolha outra letra."
                status_color = (0, 0, 255) # Vermelho
        
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # --- MUDANÇA: Mensagem de "Ocioso" dividida em duas linhas ---
    else:
        status_text_line1 = "Ocioso: Pressione uma letra (A-Z)"
        status_text_line2 = "ou 0 para iniciar."
        cv2.putText(frame, status_text_line1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.putText(frame, status_text_line2, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    
    cv2.imshow("Ferramenta de Captura - LIBRAS", frame)

    # --- Lógica de Controle por Teclado ---
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    
    elif key == 32:
        if current_label and img_count < CAPTURE_LIMIT:
            capturing = not capturing

    elif 65 <= key <= 90 or 97 <= key <= 122 or key == ord('0'):
        if key == ord('0'):
            new_label = "nenhum"
        else:
            new_label = chr(key).upper()

        current_label = new_label
        save_dir = os.path.join(DATA_DIR, current_label.upper())
        os.makedirs(save_dir, exist_ok=True)
        img_count = len(os.listdir(save_dir))
        
        if img_count < CAPTURE_LIMIT:
            capturing = True
            print(f"\n[INFO] Iniciando/Retomando captura para a classe '{current_label.upper()}'. Imagens existentes: {img_count}")
        else:
            capturing = False
            print(f"\n[AVISO] A classe '{current_label.upper()}' já atingiu o limite de {CAPTURE_LIMIT} imagens.")

# --- Finalização ---
print("[INFO] Encerrando aplicação...")
cap.release()
cv2.destroyAllWindows()