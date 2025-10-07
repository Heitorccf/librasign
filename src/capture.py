# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import os
import numpy as np
import csv

# Diretório de saída para os dados de landmarks.
DATA_DIR = "data/landmarks"
# Limite de amostras por classe para evitar desbalanceamento.
CAPTURE_LIMIT = 1000

# Inicializando a solução MediaPipe Hands com parâmetros de confiança.
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
# Obtendo o utilitário de desenho para visualização.
mp_drawing = mp.solutions.drawing_utils

# Inicializando a captura de vídeo a partir da webcam padrão.
cap = cv2.VideoCapture(0)

# Verificando a conexão com a câmera.
if not cap.isOpened():
    print("[ERRO] Não foi possível abrir a câmera, verifique a conexão e as permissões")
    exit()

# Criando a janela de visualização de forma explícita.
WINDOW_NAME = "Captura de Landmarks - LIBRAS"
cv2.namedWindow(WINDOW_NAME)

# Variáveis de estado para o fluxo de captura.
is_capturing = False
current_label = None
capture_count = 0

# Exibindo as instruções de uso no console.
print("-" * 50)
print("Ferramenta de Captura de Landmarks - LIBRAS")
print("INSTRUÇÕES:")
print(" > Pressione uma letra (A-Z) ou 0 para iniciar a captura")
print(" > Pressione 'ESPACO' para pausar ou retomar")
print(" > Pressione 'ESC' para sair")
print("-" * 50)


while True:
    # Lendo um novo frame da câmera.
    ret, frame = cap.read()
    if not ret:
        print("[AVISO] Frame da câmera não pôde ser lido, encerrando")
        break

    # Espelhando o frame horizontalmente para efeito de espelho.
    frame = cv2.flip(frame, 1)

    # Convertendo o frame de BGR para RGB, formato esperado pelo MediaPipe.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Otimizando o desempenho ao processar a imagem por referência.
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    # Verificando se landmarks de mão foram detectados.
    if results.multi_hand_landmarks:
        # Acessando o primeiro conjunto de landmarks detectado.
        hand_landmarks = results.multi_hand_landmarks[0]

        # Renderizando os landmarks e conexões sobre o frame para feedback visual.
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Verificando se a captura está ativa e dentro do limite.
        if is_capturing and current_label and capture_count < CAPTURE_LIMIT:
            try:
                # Extraindo e achatando as coordenadas (x, y, z) dos 21 landmarks.
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                # Garantindo a existência do diretório de dados.
                os.makedirs(DATA_DIR, exist_ok=True)
                csv_path = os.path.join(DATA_DIR, f"{current_label}.csv")

                # Abrindo o arquivo CSV em modo 'append' para adicionar novas amostras.
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    # Escrevendo o vetor de coordenadas.
                    writer.writerow(coords)

                # Incrementando o contador de amostras da classe.
                capture_count += 1
            except Exception as e:
                print(f"[ERRO] Falha ao salvar os dados: {e}")

    # Gerenciando dinamicamente o texto de status para o usuário.
    if current_label:
        if capture_count >= CAPTURE_LIMIT:
            is_capturing = False
            status_text = f"CLASSE '{current_label}' COMPLETA ({capture_count}/{CAPTURE_LIMIT})"
        elif is_capturing:
            status_text = f"GRAVANDO '{current_label}' ({capture_count}/{CAPTURE_LIMIT})"
        else:
            status_text = f"PAUSADO '{current_label}' ({capture_count}/{CAPTURE_LIMIT})"
    else:
        status_text = "Ocioso, escolha uma letra para iniciar"

    # Desenhando o texto de status sobre o frame.
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # Atualizando a janela com o frame processado.
    cv2.imshow(WINDOW_NAME, frame)

    # Aguardando por uma tecla pressionada.
    key = cv2.waitKey(5) & 0xFF

    # Tecla ESC para sair.
    if key == 27:
        break
    # Tecla ESPAÇO para pausar ou retomar.
    elif key == 32:
        if current_label:
            # Alternando o estado de captura.
            is_capturing = not is_capturing
    # Tecla de letra (A-Z) ou '0' para iniciar a captura.
    elif (ord('a') <= key <= ord('z')) or (ord('A') <= key <= ord('Z')) or (key == ord('0')):
        label_name = "nenhum" if key == ord('0') else chr(key).upper()
        current_label = label_name
        csv_path = os.path.join(DATA_DIR, f"{current_label}.csv")

        # Verificando a contagem de amostras existentes para a classe.
        try:
            with open(csv_path, 'r') as f:
                capture_count = sum(1 for _ in f)
        except FileNotFoundError:
            capture_count = 0

        if capture_count < CAPTURE_LIMIT:
            is_capturing = True
            print(f"\n[INFO] Iniciando captura para '{current_label}', capturas existentes: {capture_count}")
        else:
            is_capturing = False
            print(f"\n[AVISO] Classe '{current_label}' já atingiu o limite de {CAPTURE_LIMIT} amostras")

# Liberando os recursos ao final da execução.
print("[INFO] Encerrando aplicação")
hands.close()
cap.release()
cv2.destroyAllWindows()