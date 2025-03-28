import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from src.normalizing import load_and_preprocess_images

# Carrega as imagens e rótulos
print("[INFO] Carregando e processando imagens...")
X, y = load_and_preprocess_images()

# Codifica os rótulos em one-hot (ex: A -> [1, 0, 0, ...])
encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(y)

# Divide os dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Define o modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_encoded.shape[1], activation='softmax')  # uma saída por letra
])

# Compila o modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Salva o melhor modelo durante o treino
checkpoint = ModelCheckpoint(
    "models/best_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1
)

# Treina o modelo
print("[INFO] Treinando modelo...")
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

print("[INFO] Treinamento finalizado. Modelo salvo em models/best_model.keras")