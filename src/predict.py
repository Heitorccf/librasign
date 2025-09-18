# -*- coding: utf-8 -*-
"""
Sistema de Inferência em Tempo Real para Reconhecimento de Gestos em LIBRAS

Este módulo está implementando a aplicação do modelo treinado, incluindo:
1, Normalização de landmarks em tempo real para robustez
2, Filtro de suavização temporal para estabilizar as predições
3, Exibição da confiança do modelo para feedback ao usuário
4, Lógica de confirmação por tempo para construção de frases
"""
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque

# ============================================================================
# Bloco de Funções e Configuração Inicial
# ============================================================================

def normalize_landmarks(data):
    """
    Normaliza um único conjunto de 63 coordenadas de landmarks

    O motivo desta função é garantir que a entrada do modelo seja invariante
    à posição e à escala da mão, tornando a predição mais robusta
    Ela está replicando o exato mesmo pré-processamento usado no treinamento
    """
    landmarks = data.reshape((21, 3)) # Remodelando para 21 pontos (x, y, z)

    # Invariância à Posição: Centrando todos os pontos em relação ao pulso
    base_point = landmarks[0].copy()
    relative_landmarks = landmarks - base_point

    # Invariância à Escala: Normalizando pela distância entre pulso e base do dedo médio
    scale_dist = np.linalg.norm(relative_landmarks[9])
    if scale_dist < 1e-6: # Evitando uma potencial divisão por zero
        scale_dist = 1.0

    scaled_landmarks = relative_landmarks / scale_dist
    return scaled_landmarks.flatten()

# Carregando os artefatos do modelo previamente treinados
# Esta abordagem permite que a aplicação opere sem a necessidade de retreinar
print("[INFO] Carregando modelo MLP, normalizador e classes")
with open("models/librasign_mlp.pkl", 'rb') as f:
    model = pickle.load(f)
with open("models/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

class_names = np.load("models/classes.npy")

# Inicializando a solução MediaPipe Hands
hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils # Obtendo o utilitário de desenho

# Configurando os parâmetros para a lógica de inferência
HISTORY_SIZE = 10 # Usando os últimos 10 frames para estabilizar a predição
CONFIDENCE_THRESHOLD = 0.75 # Exigindo 75% de confiança para considerar uma predição válida
predictions_history = deque(maxlen=HISTORY_SIZE)

# Configurando os parâmetros para a construção de frases
sentence = []
CONFIRMATION_TIME = 2.0 # Definindo 2 segundos de estabilidade para confirmar uma letra
stable_letter_start_time = None
current_stable_letter = ""
letter_confirmed = False

# Inicializando a captura de vídeo
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRO] Não foi possível abrir a câmera, verifique a conexão")
    exit()

# Criando a janela de visualização explicitamente antes do loop
# O motivo é garantir um comportamento estável da GUI em diferentes sistemas
WINDOW_NAME = "Librasign - Tradutor de LIBRAS"
cv2.namedWindow(WINDOW_NAME)

print("[INFO] Sistema pronto, pressione 'ESC' para sair")

# ============================================================================
# Bloco do Loop Principal de Execução
# ============================================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    H, W, _ = frame.shape
    frame = cv2.flip(frame, 1) # Espelhando o frame para uma visualização intuitiva
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertendo para o formato do MediaPipe

    # Otimizando o desempenho ao passar a imagem como não gravável
    image_rgb.flags.writeable = False
    result = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    # Verificando se landmarks de mão foram detectados
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
        # ============================================================================
        # Pipeline de Pré-processamento e Predição em Tempo Real
        # ============================================================================
        
        # 1, Extraindo as coordenadas brutas
        coords_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        
        # 2, Aplicando a normalização de pose e escala
        coords_normalized = normalize_landmarks(coords_raw).reshape(1, -1)
        
        # 3, Aplicando o StandardScaler carregado
        scaled_coords = scaler.transform(coords_normalized)
        
        # Obtendo as probabilidades de cada classe
        # A escolha por 'predict_proba' ao invés de 'predict' permite
        # implementar uma lógica baseada em confiança
        prediction_proba = model.predict_proba(scaled_coords)
        prediction_index = np.argmax(prediction_proba)
        confidence = prediction_proba[0][prediction_index]
        
        # Adicionando a predição ao histórico apenas se a confiança for suficiente
        if confidence >= CONFIDENCE_THRESHOLD:
            predictions_history.append(class_names[prediction_index])
        else:
            predictions_history.append(None) # Usando None para predições incertas
    else:
        # Limpando o histórico se nenhuma mão for detectada
        # Isso força o sistema a resetar a letra estável
        predictions_history.clear()

    # ============================================================================
    # Bloco de Lógica de Suavização e Construção de Frase
    # ============================================================================

    # Aplicando um filtro de votação majoritária para suavizar a predição
    # O objetivo é evitar que a saída visual oscile devido a pequenas variações no gesto
    valid_preds = [p for p in predictions_history if p is not None]
    smoothed_label = max(set(valid_preds), key=valid_preds.count) if valid_preds else ""

    # Implementando a lógica de confirmação por tempo de permanência
    if smoothed_label and smoothed_label != current_stable_letter:
        # Se uma nova letra estável aparece, o cronômetro é iniciado
        current_stable_letter = smoothed_label
        stable_letter_start_time = time.time()
        letter_confirmed = False
    elif smoothed_label and smoothed_label == current_stable_letter:
        # Se a letra continua estável, o tempo decorrido é verificado
        if not letter_confirmed and (time.time() - stable_letter_start_time) >= CONFIRMATION_TIME:
            sentence.append(current_stable_letter) # Adicionando a letra à frase
            letter_confirmed = True # Marcando como confirmada para evitar repetições
    elif not smoothed_label:
        # Resetando o estado se nenhuma letra estável for detectada
        current_stable_letter = ""
        stable_letter_start_time = None
        letter_confirmed = False

    # ============================================================================
    # Bloco de Interface e Feedback Visual
    # ============================================================================
    
    # Gerenciando o texto de status principal
    if current_stable_letter:
        display_text = f"Gesto: {current_stable_letter}"
        if letter_confirmed:
            display_text += " (Confirmado)"
        elif stable_letter_start_time:
            # Desenhando uma barra de progresso para a confirmação
            progress = (time.time() - stable_letter_start_time) / CONFIRMATION_TIME
            cv2.rectangle(frame, (10, 70), (10 + int(progress * 200), 90), (0, 255, 0), -1)
    else:
        display_text = "Aguardando gesto"
    
    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    
    # Desenhando a frase formada na parte inferior da tela
    cv2.rectangle(frame, (0, H - 60), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, " ".join(sentence), (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    
    cv2.imshow(WINDOW_NAME, frame)
    
    # ============================================================================
    # Bloco de Controle via Teclado
    # ============================================================================
    
    key = cv2.waitKey(5) & 0xFF
    if key == 27: # Tecla ESC para sair
        break
    if key == 8: # Tecla Backspace para apagar
        if sentence:
            sentence.pop()
    if key == ord('c'): # Tecla 'c' para limpar a frase
        sentence.clear()

# ============================================================================
# Bloco de Finalização
# ============================================================================

print("[INFO] Encerrando aplicação")
cap.release()
cv2.destroyAllWindows()