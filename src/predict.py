# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque

def normalize_landmarks(data):
    """
    Normalizando um único conjunto de 63 coordenadas de landmarks,
    replicando o pré-processamento usado no treinamento para garantir
    invariância à posição e à escala da mão.
    """
    # Remodelando para 21 pontos (x, y, z).
    landmarks = data.reshape((21, 3))

    # Centralizando os pontos em relação ao pulso.
    base_point = landmarks[0].copy()
    relative_landmarks = landmarks - base_point

    # Normalizando pela distância entre o pulso e a base do dedo médio.
    scale_dist = np.linalg.norm(relative_landmarks[9])
    if scale_dist < 1e-6: # Evitando divisão por zero.
        scale_dist = 1.0

    scaled_landmarks = relative_landmarks / scale_dist
    return scaled_landmarks.flatten()

# Carregando os artefatos do modelo previamente treinado.
print("[INFO] Carregando modelo MLP, normalizador e classes")
with open("models/librasign_mlp.pkl", 'rb') as f:
    model = pickle.load(f)
with open("models/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

class_names = np.load("models/classes.npy")

# Inicializando a solução MediaPipe Hands.
hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
# Obtendo o utilitário de desenho.
mp_draw = mp.solutions.drawing_utils

# Parâmetros para a lógica de inferência.
HISTORY_SIZE = 10           # Usando os últimos 10 frames para estabilizar a predição.
CONFIDENCE_THRESHOLD = 0.75 # Confiança mínima para uma predição ser considerada válida.
predictions_history = deque(maxlen=HISTORY_SIZE)

# Parâmetros para a construção de frases.
sentence = []
CONFIRMATION_TIME = 2.0 # Tempo de estabilidade para confirmar uma letra.
stable_letter_start_time = None
current_stable_letter = ""
letter_confirmed = False

# Inicializando a captura de vídeo.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRO] Não foi possível abrir a câmera, verifique a conexão")
    exit()

# Criando a janela de visualização de forma explícita.
WINDOW_NAME = "Librasign - Tradutor de LIBRAS"
cv2.namedWindow(WINDOW_NAME)

print("[INFO] Sistema pronto, pressione 'ESC' para sair")


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    H, W, _ = frame.shape
    # Espelhando o frame para uma visualização intuitiva.
    frame = cv2.flip(frame, 1)
    # Convertendo para o formato RGB do MediaPipe.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Otimizando o desempenho ao passar a imagem como não gravável.
    image_rgb.flags.writeable = False
    result = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    # Verificando se landmarks de mão foram detectados.
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
        # 1. Extraindo as coordenadas brutas.
        coords_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        
        # 2. Aplicando a normalização de pose e escala.
        coords_normalized = normalize_landmarks(coords_raw).reshape(1, -1)
        
        # 3. Aplicando o StandardScaler carregado.
        scaled_coords = scaler.transform(coords_normalized)
        
        # Obtendo as probabilidades de cada classe.
        prediction_proba = model.predict_proba(scaled_coords)
        prediction_index = np.argmax(prediction_proba)
        confidence = prediction_proba[0][prediction_index]
        
        # Adicionando a predição ao histórico se a confiança for suficiente.
        if confidence >= CONFIDENCE_THRESHOLD:
            predictions_history.append(class_names[prediction_index])
        else:
            # Usando None para predições incertas.
            predictions_history.append(None)
    else:
        # Limpando o histórico se nenhuma mão for detectada.
        predictions_history.clear()

    # Aplicando um filtro de votação majoritária para suavizar a predição.
    valid_preds = [p for p in predictions_history if p is not None]
    smoothed_label = max(set(valid_preds), key=valid_preds.count) if valid_preds else ""

    # Implementando a lógica de confirmação por tempo de permanência.
    if smoothed_label and smoothed_label != current_stable_letter:
        # Iniciando o cronômetro para uma nova letra estável.
        current_stable_letter = smoothed_label
        stable_letter_start_time = time.time()
        letter_confirmed = False
    elif smoothed_label and smoothed_label == current_stable_letter:
        # Verificando se a letra permaneceu estável pelo tempo necessário.
        if not letter_confirmed and (time.time() - stable_letter_start_time) >= CONFIRMATION_TIME:
            # Adicionando a letra à frase.
            sentence.append(current_stable_letter)
            # Marcando como confirmada para evitar repetições.
            letter_confirmed = True
    elif not smoothed_label:
        # Resetando o estado se nenhuma letra estável for detectada.
        current_stable_letter = ""
        stable_letter_start_time = None
        letter_confirmed = False
    
    # Gerenciando o texto de status principal.
    if current_stable_letter:
        display_text = f"Gesto: {current_stable_letter}"
        if letter_confirmed:
            display_text += " (Confirmado)"
        elif stable_letter_start_time:
            # Desenhando uma barra de progresso para a confirmação.
            progress = (time.time() - stable_letter_start_time) / CONFIRMATION_TIME
            cv2.rectangle(frame, (10, 70), (10 + int(progress * 200), 90), (0, 255, 0), -1)
    else:
        display_text = "Aguardando gesto"
    
    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    
    # Desenhando a frase formada na parte inferior da tela.
    cv2.rectangle(frame, (0, H - 60), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, " ".join(sentence), (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    
    cv2.imshow(WINDOW_NAME, frame)
    
    key = cv2.waitKey(5) & 0xFF
    # Tecla ESC para sair.
    if key == 27:
        break
    # Tecla Backspace para apagar.
    if key == 8:
        if sentence:
            sentence.pop()
    # Tecla 'c' para limpar a frase.
    if key == ord('c'):
        sentence.clear()

# Liberando os recursos ao final da execução.
print("[INFO] Encerrando aplicação")
cap.release()
cv2.destroyAllWindows()