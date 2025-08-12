# -*- coding: utf-8 -*-
"""
Script para Captura de Dados Visuais para Treinamento de Modelo.

Este módulo é responsável pela aquisição de imagens da Língua Brasileira
de Sinais (Libras) através de uma webcam. Ele utiliza a biblioteca OpenCV
para a interface com a câmera e o MediaPipe para a detecção de mãos em
tempo real, recortando e salvando a região de interesse para posterior
processamento e treinamento da rede neural.
"""

import cv2
import mediapipe as mp
import os
from datetime import datetime

# --- Inicialização de Componentes Essenciais ---

# Inicializa a solução 'Hands' da biblioteca MediaPipe para a detecção de
# marcos de referência (landmarks) da mão.
# O parâmetro 'max_num_hands=1' restringe a detecção a uma única mão,
# otimizando o desempenho e focando no escopo do projeto.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Inicializa as utilidades de desenho do MediaPipe para visualizar
# os landmarks e as conexões da mão sobre a imagem.
mp_draw = mp.solutions.drawing_utils

# --- Configuração de Parâmetros ---

# Define o diretório raiz para o armazenamento dos dados brutos.
# As imagens capturadas serão organizadas em subdiretórios nomeados
# de acordo com a sua classe (letra do alfabeto).
DATA_DIR = "data/raw"

# Define a resolução padrão (altura e largura) para as imagens salvas.
# A uniformidade de tamanho é um pré-requisito para o treinamento de
# Redes Neurais Convolucionais (CNNs).
IMG_SIZE = 224

# --- Loop Principal de Captura ---

# Instancia um objeto VideoCapture para acessar o stream da webcam primária (índice 0).
cap = cv2.VideoCapture(0)

print("[INFO] Pressione uma tecla (A-Z) para iniciar a captura de imagens para essa letra.")
print("[INFO] Pressione a tecla 'ESC' para finalizar a execução.")

current_label = None  # Variável para armazenar o rótulo (letra) da captura atual.
img_count = 0         # Contador para o número de imagens capturadas por rótulo.

while True:
    # Realiza a leitura de um frame do vídeo. 'ret' é um booleano que indica
    # sucesso na captura, e 'frame' é a matriz da imagem.
    ret, frame = cap.read()
    if not ret:
        print("[AVISO] Não foi possível capturar o frame. Encerrando.")
        break

    # Inverte o frame horizontalmente (espelhamento).
    # Esta operação proporciona uma experiência de usuário mais intuitiva.
    frame = cv2.flip(frame, 1)

    # Converte o frame do espaço de cores BGR (padrão do OpenCV) para RGB.
    # O MediaPipe espera imagens no formato RGB para o processamento.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa o frame RGB para detectar a presença de mãos.
    result = hands.process(frame_rgb)

    # Verifica se foram detectados landmarks de mão no frame.
    if result.multi_hand_landmarks:
        # Itera sobre cada mão detectada (neste caso, no máximo uma).
        for hand_landmarks in result.multi_hand_landmarks:
            # Renderiza os landmarks e as conexões sobre o frame original (colorido).
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtém as dimensões do frame para converter as coordenadas
            # normalizadas (0 a 1) dos landmarks em coordenadas de pixel.
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            # Calcula a Bounding Box (caixa delimitadora) que envolve a mão.
            # Uma margem de 20 pixels é adicionada para garantir que a mão inteira
            # seja capturada, mesmo em movimentos rápidos.
            x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
            y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20

            # Garante que as coordenadas da Bounding Box não excedam os limites do frame.
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            # Recorta a Região de Interesse (ROI - Region of Interest) do frame original.
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Assegura que a ROI não está vazia antes de prosseguir.
            if hand_img.size == 0:
                continue

            # Converte a ROI recortada para escala de cinza. Esta é uma etapa crucial
            # de otimização, pois remove informações de cor irrelevantes (tons de pele,
            # iluminação) e foca o aprendizado do modelo na forma do gesto.
            gray_hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

            # Redimensiona a imagem em escala de cinza para o tamanho padrão (IMG_SIZE).
            gray_hand_img_resized = cv2.resize(gray_hand_img, (IMG_SIZE, IMG_SIZE))

            # Se uma letra foi selecionada pelo usuário, procede com o salvamento.
            if current_label:
                # Define o caminho completo do diretório para a letra atual.
                save_dir = os.path.join(DATA_DIR, current_label.upper())
                # Cria o diretório se ele não existir.
                os.makedirs(save_dir, exist_ok=True)

                # Gera um nome de arquivo único utilizando um timestamp preciso.
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                filename = f"{current_label.upper()}_{timestamp}.jpg"

                # Salva a imagem processada (em escala de cinza e redimensionada) no disco.
                cv2.imwrite(os.path.join(save_dir, filename), gray_hand_img_resized)
                img_count += 1

                # Exibe um feedback visual na tela indicando o processo de captura.
                cv2.putText(frame, f"Salvando {current_label.upper()} - {img_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibe a janela com o vídeo da webcam e as sobreposições.
    cv2.imshow("Captura de Gestos - LIBRAS", frame)

    # Aguarda por uma tecla pressionada (1 milissegundo de espera).
    key = cv2.waitKey(1) & 0xFF

    # Condição de parada: se a tecla 'ESC' (código ASCII 27) for pressionada.
    if key == 27:
        break
    # Condição de captura: se uma tecla de 'A' a 'Z' for pressionada.
    elif 65 <= key <= 90 or 97 <= key <= 122:
        current_label = chr(key).upper()
        img_count = 0  # Reinicia o contador para a nova letra.
        print(f"[INFO] Capturando imagens para a letra: {current_label}")

# --- Finalização ---

# Libera o dispositivo da webcam.
cap.release()
# Fecha todas as janelas criadas pelo OpenCV.
cv2.destroyAllWindows()