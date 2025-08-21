# -*- coding: utf-8 -*-
"""
Sistema de Treinamento Avançado com Fine-Tuning em Duas Etapas.

Este módulo implementa o pipeline de treinamento de ponta a ponta,
utilizando o MobileNetV2 como modelo base. Aplica augmentation de dados
e uma estratégia de fine-tuning em duas fases para maximizar a capacidade
de generalização do modelo.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- Etapa 1: Configuração e Geração de Dados ---
print("[INFO] Configurando geradores de imagem...")

DATA_DIR = "data/raw"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation'
)

NUM_CLASSES = train_generator.num_classes

# --- Etapa 2: Construção do Modelo ---
print(f"[INFO] Construindo modelo com base no MobileNetV2...")

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# --- Etapa 3.1: Treinamento da Cabeça ---
print("\n[FASE 1] Treinando apenas a cabeça de classificação...")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Salva o modelo intermediário (apenas com a "cabeça" treinada)
checkpoint_head = ModelCheckpoint(
    "models/librasign_head.keras", # Nome intermediário simplificado
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

history_head = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[checkpoint_head, early_stopping]
)

# --- Etapa 3.2: Fine-Tuning do Modelo Completo ---
print("\n[FASE 2] Realizando fine-tuning das camadas superiores...")

# Carrega o melhor modelo da Fase 1
model.load_weights("models/librasign_head.keras")

# Descongela o modelo base
base_model.trainable = True
print(f"[INFO] Modelo base descongelado. Número de camadas treináveis: {len(model.trainable_variables)}")

# Recompila com uma taxa de aprendizado muito baixa
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- MUDANÇA: Nome final do modelo alterado ---
checkpoint_fine_tune = ModelCheckpoint(
    "models/librasign.keras", # Nome final e simplificado
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

history_fine_tune = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[checkpoint_fine_tune, early_stopping, reduce_lr]
)

print("[INFO] Treinamento finalizado. Melhor modelo salvo em 'models/librasign.keras'.")