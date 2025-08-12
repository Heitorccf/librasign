# -*- coding: utf-8 -*-
"""
Script para Predição em Tempo Real com o Modelo Treinado.

Este módulo implementa a aplicação final. Ele carrega o modelo de CNN
previamente treinado, captura o vídeo da webcam, detecta a mão do usuário,
processa a imagem da mão em tempo real e utiliza o modelo para prever
a qual letra do alfabeto de Libras o gesto corresponde, exibindo o
resultado em uma sobreposição na tela.
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import os

print("[DEBUG] Script de predição iniciado.")

# Desabilita otimizações do oneDNN que podem causar inconsistências numéricas.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Etapa 1: Carregamento de Recursos ---

MODEL_PATH = "models/best_model.keras"  # Caminho para o modelo treinado.
IMG_SIZE = 224                          # Tamanho das imagens de entrada do modelo.

print(f"[INFO] Carregando modelo de: {MODEL_PATH}")
# Carrega a arquitetura e os pesos do modelo treinado a partir do arquivo .keras.
model = load_model(MODEL_PATH)
print("[DEBUG] Modelo carregado com sucesso.")

# Para decodificar as predições do modelo (que são em formato one-hot),
# é necessário recriar o mapeamento entre índices e rótulos (letras).
labels_path = "data/raw"
if not os.path.exists(labels_path) or not os.listdir(labels_path):
    print(f"[ERRO CRÍTICO] Diretório de dados '{labels_path}' não encontrado ou vazio. "
          "Execute os scripts 'capture.py' e 'train.py' primeiro.")
    exit()

# O LabelBinarizer é reajustado com os nomes dos diretórios (que são as classes)
# para garantir que a correspondência de rótulos seja a mesma do treinamento.
labels = sorted(os.listdir(labels_path))
encoder = LabelBinarizer()
encoder.fit(labels)
print("[DEBUG] Rótulos (labels) processados e prontos para decodificação.")

# Inicializa o MediaPipe Hands com parâmetros de alta confiança para evitar
# falsos positivos durante a predição em tempo real.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Etapa 2: Loop Principal de Predição ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRO CRÍTICO] Falha ao acessar a webcam. "
          "Verifique se ela não está em uso por outro aplicativo.")
    exit()

print("[INFO] Sistema pronto. Pressione 'ESC' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[AVISO] Falha ao capturar frame da webcam.")
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    prediction_label = ""  # Inicializa a variável de predição.

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # O recorte da mão segue a mesma lógica do script de captura.
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
                # O pré-processamento da ROI deve espelhar exatamente as etapas
                # realizadas no script `normalizing.py` antes do treinamento.
                gray_hand = cv2.cvtColor(hand_img_roi, cv2.COLOR_BGR2GRAY)
                hand_img_resized = cv2.resize(gray_hand, (IMG_SIZE, IMG_SIZE))
                hand_img_normalized = hand_img_resized.astype(np.float32) / 255.0

                # A imagem precisa ser expandida para o formato 4D esperado pelo modelo.
                # A dimensão do batch (lote) é 1, pois estamos prevendo uma única imagem.
                # Formato final: (1, 224, 224, 1).
                hand_img_expanded = np.expand_dims(hand_img_normalized, axis=0)
                hand_img_expanded = np.expand_dims(hand_img_expanded, axis=-1)

                # Realiza a inferência. O modelo retorna um array de probabilidades.
                prediction = model.predict(hand_img_expanded, verbose=0)
                # `np.argmax` encontra o índice do neurônio com a maior probabilidade.
                predicted_index = np.argmax(prediction)
                # Usa o `encoder` para converter o índice de volta para o rótulo textual.
                prediction_label = encoder.classes_[predicted_index]

    # --- Etapa 3: Visualização da Interface Gráfica (UI) ---

    # Esta seção é dedicada a criar um display informativo e esteticamente
    # agradável para o usuário, mostrando a predição em tempo real.
    display_text = f"Letra: {prediction_label}" if prediction_label else "Aguardando gesto..."
    bg_color = (0, 128, 0) if prediction_label else (128, 0, 0) # Verde para predição, Vermelho para espera

    # Calcula o tamanho do texto para desenhar uma caixa de fundo.
    (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_TRIPLEX, 1.0, 2)
    
    # Define as coordenadas para o retângulo e o texto.
    rect_start = (frame.shape[1] - text_width - 30, 10)
    rect_end = (frame.shape[1] - 10, 20 + text_height)
    text_pos = (frame.shape[1] - text_width - 20, 10 + text_height)

    # Desenha um retângulo com transparência para destacar o texto.
    overlay = frame.copy()
    cv2.rectangle(overlay, rect_start, rect_end, bg_color, cv2.FILLED)
    alpha = 0.6  # Nível de transparência.
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Escreve o texto da predição sobre o retângulo.
    cv2.putText(frame, display_text, text_pos, cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Reconhecimento de Gestos - LIBRAS", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para sair.
        break

# --- Finalização ---
print("[DEBUG] Saindo do loop principal e liberando recursos.")
cap.release()
cv2.destroyAllWindows()
print("[DEBUG] Fim do script.")