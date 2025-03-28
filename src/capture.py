import cv2
import mediapipe as mp
import os
from datetime import datetime

# Inicializa o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Define o caminho onde as imagens serão salvas
DATA_DIR = "data/raw"

# Tamanho da imagem salva
IMG_SIZE = 224

# Inicia a webcam
cap = cv2.VideoCapture(0)

print("[INFO] Pressione uma tecla (A-Z) para capturar imagens dessa letra.")
print("[INFO] Pressione 'ESC' para sair.")

current_label = None  # Letra atual sendo capturada
img_count = 0         # Contador de imagens por letra

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inverte a imagem para parecer com um espelho
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta as mãos
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Desenha os pontos da mão
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extrai a região da mão com base nos pontos
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            # Calcula a bounding box da mão
            x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
            y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20

            # Garante que os limites estejam dentro do frame
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            # Recorta e redimensiona a imagem da mão
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue
            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))

            # Salva a imagem se uma letra foi selecionada
            if current_label:
                save_dir = os.path.join(DATA_DIR, current_label.upper())
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                filename = f"{current_label.upper()}_{timestamp}.jpg"
                cv2.imwrite(os.path.join(save_dir, filename), hand_img)
                img_count += 1

                # Mostra info na tela
                cv2.putText(frame, f"Salvando {current_label.upper()} - {img_count}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibe o vídeo com sobreposições
    cv2.imshow("Captura de Gestos - LIBRAS", frame)

    # Captura a tecla pressionada
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Tecla ESC
        break
    elif 65 <= key <= 90 or 97 <= key <= 122:
        # Tecla A-Z (maiúscula ou minúscula)
        current_label = chr(key).upper()
        img_count = 0
        print(f"[INFO] Capturando imagens para a letra: {current_label}")

# Libera a webcam e fecha as janelas
cap.release()
cv2.destroyAllWindows()