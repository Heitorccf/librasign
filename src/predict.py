# -*- coding: utf-8 -*-
"""
Sistema de Reconhecimento e Predição de Gestos em Tempo Real.

Este módulo constitui a aplicação final do sistema de reconhecimento de Libras,
integrando o modelo de rede neural convolucional previamente treinado com um
pipeline de captura e processamento de vídeo. O sistema realiza a detecção
contínua de gestos manuais através da webcam, processando cada quadro em
tempo real e classificando o gesto capturado em sua respectiva letra do
alfabeto, apresentando os resultados através de uma interface visual interativa.
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import os

print("[DEBUG] Script de predição iniciado.")

# Desabilitando otimizações do oneDNN para garantir consistência e
# reprodutibilidade nos cálculos numéricos durante a inferência.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Inicializando recursos e configurações do sistema

MODEL_PATH = "models/best_model.keras"  # Especificando o caminho do modelo pré-treinado.
IMG_SIZE = 224                          # Definindo dimensões padronizadas para entrada do modelo.

print(f"[INFO] Carregando modelo de: {MODEL_PATH}")
# Restaurando a arquitetura completa e os parâmetros aprendidos do modelo neural.
model = load_model(MODEL_PATH)
print("[DEBUG] Modelo carregado com sucesso.")

# Reconstruindo o mapeamento categórico para decodificação das predições,
# garantindo correspondência exata com o esquema utilizado durante o treinamento.
labels_path = "data/raw"
if not os.path.exists(labels_path) or not os.listdir(labels_path):
    print(f"[ERRO CRÍTICO] Diretório de dados '{labels_path}' não encontrado ou vazio. "
          "Execute os scripts 'capture.py' e 'train.py' primeiro.")
    exit()

# Sincronizando o codificador com a estrutura de classes original,
# preservando a ordenação alfabética para consistência na decodificação.
labels = sorted(os.listdir(labels_path))
encoder = LabelBinarizer()
encoder.fit(labels)
print("[DEBUG] Rótulos (labels) processados e prontos para decodificação.")

# Configurando o detector de mãos com parâmetros otimizados para
# minimizar detecções espúrias e maximizar a estabilidade do rastreamento.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Executando o ciclo principal de captura e predição

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
    prediction_label = ""  # Inicializando variável para armazenar a predição atual.

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculando os limites espaciais da região contendo a mão detectada,
            # mantendo consistência com o protocolo de captura original.
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
                # Aplicando pipeline de pré-processamento idêntico ao utilizado
                # durante a fase de treinamento, garantindo consistência na inferência.
                gray_hand = cv2.cvtColor(hand_img_roi, cv2.COLOR_BGR2GRAY)
                hand_img_resized = cv2.resize(gray_hand, (IMG_SIZE, IMG_SIZE))
                hand_img_normalized = hand_img_resized.astype(np.float32) / 255.0

                # Reestruturando o tensor para conformidade com a arquitetura do modelo.
                # Adicionando dimensão de lote unitário e canal monocromático,
                # resultando no formato esperado: (1, 224, 224, 1).
                hand_img_expanded = np.expand_dims(hand_img_normalized, axis=0)
                hand_img_expanded = np.expand_dims(hand_img_expanded, axis=-1)

                # Executando inferência através da propagação direta na rede neural.
                prediction = model.predict(hand_img_expanded, verbose=0)
                # Identificando a classe de maior probabilidade através do argumento máximo.
                predicted_index = np.argmax(prediction)
                # Decodificando o índice numérico para sua representação alfabética.
                prediction_label = encoder.classes_[predicted_index]

    # Renderizando interface visual com feedback em tempo real

    # Construindo elementos visuais informativos para comunicação efetiva
    # do estado atual do sistema e resultados da classificação.
    display_text = f"Letra: {prediction_label}" if prediction_label else "Aguardando gesto..."
    bg_color = (0, 128, 0) if prediction_label else (128, 0, 0)  # Codificando estado através de cores.

    # Calculando dimensões textuais para posicionamento preciso da interface.
    (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_TRIPLEX, 1.0, 2)
    
    # Estabelecendo coordenadas para elementos gráficos de sobreposição.
    rect_start = (frame.shape[1] - text_width - 30, 10)
    rect_end = (frame.shape[1] - 10, 20 + text_height)
    text_pos = (frame.shape[1] - text_width - 20, 10 + text_height)

    # Aplicando composição alfa para transparência visual elegante.
    overlay = frame.copy()
    cv2.rectangle(overlay, rect_start, rect_end, bg_color, cv2.FILLED)
    alpha = 0.6  # Definindo grau de opacidade para o efeito de sobreposição.
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Renderizando texto de predição com alta qualidade através de antialiasing.
    cv2.putText(frame, display_text, text_pos, cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Reconhecimento de Gestos - LIBRAS", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Detectando comando de encerramento através da tecla ESC.
        break

# Finalizando execução e liberando recursos alocados
print("[DEBUG] Saindo do loop principal e liberando recursos.")
cap.release()
cv2.destroyAllWindows()
print("[DEBUG] Fim do script.")