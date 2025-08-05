import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import os

print("[DEBUG] Script iniciado.")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Configurações e Carregamento Inicial ---
MODEL_PATH = "models/best_model.keras"
IMG_SIZE = 224

print("[INFO] Carregando modelo treinado...")
model = load_model(MODEL_PATH)
print("[DEBUG] Modelo carregado com sucesso.")

# Recupera as classes (rótulos) a partir dos nomes dos diretórios em data/raw
labels_path = "data/raw"
print(f"[DEBUG] Verificando o caminho dos dados: '{labels_path}'")
if not os.path.exists(labels_path) or not os.listdir(labels_path):
    print(f"[ERRO CRÍTICO] O diretório de dados '{labels_path}' não foi encontrado ou está vazio. Por favor, execute o script 'capture.py' primeiro.")
    exit()
print("[DEBUG] Diretório de dados verificado com sucesso.")

labels = sorted(os.listdir(labels_path))
encoder = LabelBinarizer()
encoder.fit(labels)
print("[DEBUG] Rótulos (labels) processados.")

# Inicializa MediaPipe Hands para detecção
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
print("[DEBUG] MediaPipe inicializado.")

# Inicia a webcam
print("[DEBUG] Tentando iniciar a webcam (cv2.VideoCapture)...")
cap = cv2.VideoCapture(0)

# Verificação crucial se a câmera abriu
if not cap.isOpened():
    print("[ERRO CRÍTICO] FALHA AO ABRIR A WEBCAM. Verifique se a câmera não está em uso por outro programa (Zoom, Teams, etc.) ou se o dispositivo está conectado corretamente.")
    exit()
print("[DEBUG] Webcam iniciada com sucesso.")

print("[INFO] Pressione ESC para sair.")
print("[DEBUG] Entrando no loop principal...")

# --- Loop Principal de Captura e Predição ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("[AVISO] Não foi possível capturar o frame da webcam neste ciclo.")
        continue # Usa 'continue' em vez de 'break' para tentar o próximo frame

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
            
            margin = 20
            x_min, x_max = int(min(x_coords)) - margin, int(max(x_coords)) + margin
            y_min, y_max = int(min(y_coords)) - margin, int(max(y_coords)) + margin
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            hand_img_roi = frame[y_min:y_max, x_min:x_max]
            
            if hand_img_roi.size > 0:
                hand_img_processed = cv2.resize(hand_img_roi, (IMG_SIZE, IMG_SIZE))
                hand_img_processed_rgb = cv2.cvtColor(hand_img_processed, cv2.COLOR_BGR2RGB)
                hand_img_normalized = hand_img_processed_rgb.astype(np.float32) / 255.0
                hand_img_expanded = np.expand_dims(hand_img_normalized, axis=0)

                prediction = model.predict(hand_img_expanded)
                predicted_index = np.argmax(prediction)
                prediction_label = encoder.classes_[predicted_index]

    # --- Seção de Exibição da Interface (UI) ---
    display_text = ""
    text_color = (255, 255, 255)
    bg_color_default = (128, 0, 0)
    bg_color_prediction = (0, 128, 0)

    font_scale = 1.0
    font_thickness = 2
    font_face = cv2.FONT_HERSHEY_TRIPLEX 
    padding = 10

    if prediction_label:
        display_text = f"Letra: {prediction_label}"
        current_bg_color = bg_color_prediction
    else:
        display_text = "Aguardando gesto..."
        current_bg_color = bg_color_default

    (text_width, text_height), baseline = cv2.getTextSize(display_text, font_face, font_scale, font_thickness)
    
    text_x = frame.shape[1] - text_width - padding * 2 
    text_y = padding + text_height

    rect_start_point = (text_x - padding, text_y - text_height - padding + baseline)
    rect_end_point = (text_x + text_width + padding, text_y + padding)

    overlay = frame.copy()
    cv2.rectangle(overlay, rect_start_point, rect_end_point, current_bg_color, cv2.FILLED)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, display_text, (text_x, text_y), font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    cv2.imshow("Reconhecimento de Gestos - LIBRAS", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# --- Finalização ---
print("[DEBUG] Saindo do loop principal.")
cap.release()
cv2.destroyAllWindows()
print("[DEBUG] Recursos liberados. Fim do script.")