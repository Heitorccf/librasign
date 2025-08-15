# -*- coding: utf-8 -*-
"""
Sistema de Aquisição de Dados Visuais para Treinamento de Modelo de Reconhecimento.

Este módulo implementa a captura sistemática de imagens da Língua Brasileira de 
Sinais (Libras) utilizando dispositivo de câmera. O sistema emprega a biblioteca 
OpenCV para gerenciamento da interface de vídeo e o framework MediaPipe para 
detecção e rastreamento de mãos em tempo real, realizando o recorte automático 
da região de interesse e armazenamento estruturado dos dados coletados.
"""

import cv2
import mediapipe as mp
import os
from datetime import datetime

# Configurando os componentes fundamentais do sistema

# Inicializando o módulo de detecção de mãos do MediaPipe com parâmetros
# otimizados para o contexto de captura individual. A restrição para detecção
# de uma única mão maximiza a eficiência computacional e mantém o foco no
# objetivo específico do projeto.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Carregando as ferramentas de renderização visual do MediaPipe para 
# representação gráfica dos pontos de referência anatômicos detectados.
mp_draw = mp.solutions.drawing_utils

# Definindo parâmetros operacionais do sistema

# Estabelecendo o diretório base para armazenamento hierárquico dos dados.
# A estrutura de pastas seguirá o padrão de categorização por classe alfabética.
DATA_DIR = "data/raw"

# Especificando as dimensões padronizadas para as imagens processadas.
# A padronização dimensional é requisito fundamental para o treinamento
# eficaz de arquiteturas convolucionais.
IMG_SIZE = 224

# Executando o ciclo principal de aquisição de dados

# Estabelecendo conexão com o dispositivo de captura de vídeo padrão do sistema.
cap = cv2.VideoCapture(0)

print("[INFO] Pressione uma tecla (A-Z) para iniciar a captura de imagens para essa letra.")
print("[INFO] Pressione a tecla 'ESC' para finalizar a execução.")

current_label = None  # Armazenando o identificador da classe em captura.
img_count = 0         # Contabilizando as amostras coletadas por classe.

while True:
    # Capturando o quadro atual do fluxo de vídeo. O valor booleano 'ret'
    # indica o sucesso da operação, enquanto 'frame' contém os dados da imagem.
    ret, frame = cap.read()
    if not ret:
        print("[AVISO] Não foi possível capturar o frame. Encerrando.")
        break

    # Aplicando transformação de espelhamento horizontal para proporcionar
    # uma experiência mais natural ao usuário durante a interação.
    frame = cv2.flip(frame, 1)

    # Convertendo o espaço de cores de BGR (convenção OpenCV) para RGB,
    # adequando o formato aos requisitos de entrada do MediaPipe.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processando o quadro para identificação de estruturas anatômicas da mão.
    result = hands.process(frame_rgb)

    # Verificando a presença de detecções válidas no quadro processado.
    if result.multi_hand_landmarks:
        # Iterando sobre as detecções identificadas (limitadas a uma neste contexto).
        for hand_landmarks in result.multi_hand_landmarks:
            # Desenhando a representação visual dos pontos e conexões anatômicas.
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraindo as dimensões do quadro para conversão de coordenadas
            # normalizadas (intervalo [0,1]) para coordenadas absolutas em pixels.
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            # Calculando os limites da região delimitadora com margem de segurança
            # de 20 pixels, garantindo a captura completa da gesticulação mesmo
            # durante movimentos dinâmicos.
            x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
            y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20

            # Aplicando restrições aos limites para prevenir extrapolação das
            # dimensões válidas da imagem.
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            # Extraindo a região de interesse contendo a mão detectada.
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Validando a integridade da região extraída antes do processamento.
            if hand_img.size == 0:
                continue

            # Convertendo para escala de cinza, eliminando variações cromáticas
            # irrelevantes e focalizando o aprendizado nas características
            # morfológicas do gesto.
            gray_hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

            # Redimensionando a imagem para as dimensões padronizadas do conjunto de dados.
            gray_hand_img_resized = cv2.resize(gray_hand_img, (IMG_SIZE, IMG_SIZE))

            # Persistindo a amostra coletada quando uma classe está selecionada.
            if current_label:
                # Construindo o caminho completo do diretório de destino.
                save_dir = os.path.join(DATA_DIR, current_label.upper())
                # Criando a estrutura de diretórios necessária.
                os.makedirs(save_dir, exist_ok=True)

                # Gerando identificador único temporal com precisão de microssegundos.
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                filename = f"{current_label.upper()}_{timestamp}.jpg"

                # Salvando a imagem processada no sistema de arquivos.
                cv2.imwrite(os.path.join(save_dir, filename), gray_hand_img_resized)
                img_count += 1

                # Exibindo indicador visual do progresso da captura.
                cv2.putText(frame, f"Salvando {current_label.upper()} - {img_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Renderizando a interface visual com as anotações aplicadas.
    cv2.imshow("Captura de Gestos - LIBRAS", frame)

    # Capturando entrada do teclado com timeout de 1 milissegundo.
    key = cv2.waitKey(1) & 0xFF

    # Verificando condição de término através da tecla ESC (código ASCII 27).
    if key == 27:
        break
    # Detectando seleção de nova classe através de teclas alfabéticas.
    elif 65 <= key <= 90 or 97 <= key <= 122:
        current_label = chr(key).upper()
        img_count = 0  # Reinicializando o contador para a nova classe.
        print(f"[INFO] Capturando imagens para a letra: {current_label}")

# Liberando recursos do sistema

# Desconectando o dispositivo de captura de vídeo.
cap.release()
# Fechando todas as interfaces gráficas instanciadas.
cv2.destroyAllWindows()