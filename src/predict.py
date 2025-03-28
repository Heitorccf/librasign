import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import os

# Caminho para o modelo treinado
MODEL_PATH = "models/best_model.keras"
IMG_SIZE = 224

# Carrega o modelo
print("[INFO] Carregando modelo treinado...")
model = load_model(MODEL_PATH)

# Recupera as classes a partir dos diretórios usados no treino
labels = sorted(os.listdir("data/raw"))
encoder = LabelBinarizer()
encoder.fit(labels)

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Inicia a webcam
cap = cv2.VideoCapture(0)

print("[INFO] Pressione ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    prediction_label = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
            y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20

            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            hand_img = hand_img.astype(np.float32) / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Faz a previsão
            prediction = model.predict(hand_img)
            predicted_index = np.argmax(prediction)
            prediction_label = encoder.classes_[predicted_index]

            # Mostra o resultado na tela
            cv2.putText(frame, f"Letra: {prediction_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Exibe o vídeo com previsão
    cv2.imshow("Reconhecimento de Gestos - LIBRAS", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Tecla ESC
        break

cap.release()
cv2.destroyAllWindows()