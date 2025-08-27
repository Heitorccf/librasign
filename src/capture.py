# -*- coding: utf-8 -*-
"""
Ferramenta para Captura de Dados Geométricos (Landmarks) - Versão Estável

Este módulo foca na extração robusta de coordenadas 3D da mão,
salvando-as em um formato CSV. A estrutura foi simplificada para
garantir máxima estabilidade e exibir feedback visual contínuo.
"""
import cv2
import mediapipe as mp
import os
import numpy as np
import csv

# --- Configurações ---
DATA_DIR = "data/landmarks"
CAPTURE_LIMIT = 1000  # Limite de capturas por classe

# --- Inicialização de Componentes ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

# --- Variáveis de Controle de Estado ---
is_capturing = False
current_label = None
capture_count = 0

print("-" * 50)
print("Ferramenta de Captura de Landmarks - LIBRAS")
print("INSTRUÇÕES:")
print(" > Pressione uma letra (A-Z) ou 0 para iniciar a captura.")
print(" > Pressione 'ESPACO' para pausar/retomar.")
print(" > Pressione 'ESC' para sair.")
print("-" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[AVISO] Frame da câmera não pôde ser lido.")
        continue

    frame = cv2.flip(frame, 1)
    
    frame.flags.writeable = False
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    frame.flags.writeable = True

    # Lógica de captura e salvamento
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # --- MUDANÇA: O desenho agora é feito sempre que uma mão é detectada ---
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # A lógica de SALVAR os dados continua dentro da condição de captura
            if is_capturing and current_label and capture_count < CAPTURE_LIMIT:
                try:
                    landmarks = hand_landmarks.landmark
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                    
                    os.makedirs(DATA_DIR, exist_ok=True)
                    csv_path = os.path.join(DATA_DIR, f"{current_label}.csv")
                    
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(coords)
                    
                    capture_count += 1
                except Exception as e:
                    print(f"[ERRO] Falha ao salvar os dados: {e}")

    # UI para exibir o status
    status_text = "Ocioso. Escolha uma letra para iniciar."
    if current_label:
        if capture_count >= CAPTURE_LIMIT:
            is_capturing = False
            status_text = f"CLASSE '{current_label}' COMPLETA ({capture_count}/{CAPTURE_LIMIT})"
        elif is_capturing:
            status_text = f"GRAVANDO '{current_label}' ({capture_count}/{CAPTURE_LIMIT})"
        else:
            status_text = f"PAUSADO '{current_label}' ({capture_count}/{CAPTURE_LIMIT})"
    
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Captura de Landmarks - LIBRAS", frame)

    # Controle por teclado
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # ESPAÇO
        if current_label:
            is_capturing = not is_capturing
    elif 65 <= key <= 90 or 97 <= key <= 122 or key == ord('0'):
        if key == ord('0'):
            label_name = "nenhum"
        else:
            label_name = chr(key).upper()
        
        current_label = label_name
        csv_path = os.path.join(DATA_DIR, f"{current_label}.csv")
        
        try:
            with open(csv_path, 'r') as f:
                capture_count = sum(1 for row in f)
        except FileNotFoundError:
            capture_count = 0

        if capture_count < CAPTURE_LIMIT:
            is_capturing = True
            print(f"\n[INFO] Iniciando captura para '{current_label}'. Capturas existentes: {capture_count}")
        else:
            is_capturing = False # Garante que não comece a capturar se o limite já foi atingido
            print(f"\n[AVISO] Classe '{current_label}' já atingiu o limite de {CAPTURE_LIMIT}.")

# --- Finalização ---
print("[INFO] Encerrando aplicação...")
hands.close()
cap.release()
cv2.destroyAllWindows()