# -*- coding: utf-8 -*-
"""
Sistema de Treinamento Avançado com Aprendizagem por Transferência.

Este módulo utiliza um modelo pré-treinado (MobileNetV2) para extrair
características de alto nível das imagens de gestos. Apenas as camadas
superiores da rede são treinadas, resultando em um aprendizado mais
rápido e uma capacidade de generalização superior.
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall

# --- Etapa 1: Geração de Dados ---

print("[INFO] Configurando geradores de imagem para o modelo MobileNetV2...")

DATA_DIR = "data/raw"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation'
)

# --- Etapa 2: Construção do Modelo com Transfer Learning ---

print("[INFO] Construindo modelo com base no MobileNetV2...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- Etapa 3: Compilação e Treinamento ---

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

model.summary()

# --- MUDANÇA: Salvando o modelo no formato .h5 ---
checkpoint = ModelCheckpoint(
    "models/best_model_mobilenet.h5", # Alterado para .h5
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("[INFO] Iniciando o treinamento do modelo de Transfer Learning...")

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)

print("[INFO] Salvando histórico de treinamento...")
np.save('training_history_mobilenet.npy', history.history)

print("[INFO] Treinamento finalizado. Melhor modelo salvo em 'models/best_model_mobilenet.h5'.")