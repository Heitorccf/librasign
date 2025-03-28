import os
import cv2
import numpy as np

# Caminho para as imagens capturadas
DATA_DIR = "data/raw"
IMG_SIZE = 224

def load_and_preprocess_images():
    data = []
    labels = []

    # Percorre cada pasta (A, B, C...) dentro de data/raw
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)

        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            # Lê a imagem
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Redimensiona para tamanho padrão
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Converte para RGB (caso esteja em BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normaliza os pixels para ficar entre 0 e 1
            img = img.astype(np.float32) / 255.0

            # Armazena os dados e rótulos
            data.append(img)
            labels.append(label)

    # Converte para arrays numpy
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

# Teste rápido
if __name__ == "__main__":
    x, y = load_and_preprocess_images()
    print(f"Total de imagens carregadas: {len(x)}")
    print(f"Formato das imagens: {x.shape}")
    print(f"Classes encontradas: {set(y)}")