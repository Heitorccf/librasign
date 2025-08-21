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
# --- MUDANÇA: Importando camadas para o novo modelo e o MobileNetV2 ---
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall

# --- Etapa 1: Geração de Dados ---

print("[INFO] Configurando geradores de imagem para o modelo MobileNetV2...")

DATA_DIR = "data/raw"
IMG_SIZE = (224, 224) # MobileNetV2 funciona bem com essa resolução
BATCH_SIZE = 32

# Usamos a mesma augmentation de antes
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

# --- MUDANÇA CRÍTICA: color_mode agora é 'rgb' ---
# Modelos pré-treinados em ImageNet esperam 3 canais de cor (RGB).
# O gerador irá converter nossas imagens em escala de cinza para um formato
# de 3 canais, simplesmente duplicando o canal de cinza.
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb', # Alterado de 'grayscale' para 'rgb'
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb', # Alterado de 'grayscale' para 'rgb'
    class_mode='categorical',
    subset='validation'
)

# --- Etapa 2: Construção do Modelo com Transfer Learning ---

print("[INFO] Construindo modelo com base no MobileNetV2...")

# Carrega o MobileNetV2 pré-treinado com os pesos do ImageNet.
# `include_top=False` remove a camada de classificação original (que classificava 1000 objetos).
# `input_shape` deve ter 3 canais de cor.
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# "Congelamos" as camadas do modelo base. Seus pesos não serão atualizados
# durante o treinamento inicial, preservando o conhecimento que já possuem.
base_model.trainable = False

# Adicionamos nossas próprias camadas de classificação no topo do modelo base.
x = base_model.output
x = GlobalAveragePooling2D()(x) # Reduz a dimensionalidade de forma inteligente
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Este é o nosso modelo final, que combina a base MobileNetV2 com nossas camadas.
model = Model(inputs=base_model.input, outputs=predictions)

# --- Etapa 3: Compilação e Treinamento ---

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

model.summary()

checkpoint = ModelCheckpoint(
    "models/best_model_mobilenet.keras", # Novo nome para o modelo
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

print("[INFO] Treinamento finalizado. Melhor modelo salvo em 'models/best_model_mobilenet.keras'.")