# -*- coding: utf-8 -*-
"""
Script para Treinamento da Rede Neural Convolucional (CNN).

Este módulo orquestra todo o processo de treinamento do modelo.
Ele carrega os dados pré-processados, define a arquitetura da CNN,
compila o modelo com um otimizador e uma função de perda, e
executa o treinamento, salvando o melhor modelo encontrado
com base na acurácia de validação.
"""

import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from normalizing import load_and_preprocess_images  # Importa a função do módulo vizinho.

# --- Etapa 1: Carregamento e Preparação dos Dados ---

print("[INFO] Carregando e pré-processando imagens...")
X, y = load_and_preprocess_images()

# Utiliza o LabelBinarizer para converter os rótulos textuais (ex: 'A', 'B')
# em um formato de codificação one-hot (ex: A -> [1,0,0,0], B -> [0,1,0,0]).
# Este formato é o padrão para problemas de classificação multiclasse.
encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(y)

# Divide o conjunto de dados em subconjuntos de treinamento e validação.
# 80% dos dados serão usados para treinar o modelo e 20% para validar seu
# desempenho em dados não vistos. O parâmetro 'stratify' garante que a
# proporção de amostras para cada classe seja a mesma em ambos os subconjuntos.
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Etapa 2: Definição da Arquitetura do Modelo ---

model = Sequential([
    # Primeira camada convolucional: extrai 32 mapas de características
    # usando um kernel (filtro) 3x3. A 'input_shape' define o formato dos
    # dados de entrada: imagens de 224x224 com 1 canal (escala de cinza).
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    # Camada de Max Pooling: reduz a dimensionalidade espacial (downsampling)
    # pela metade, mantendo as características mais proeminentes.
    MaxPooling2D(pool_size=(2, 2)),

    # Segunda camada convolucional: aumenta a complexidade da extração para
    # 64 mapas de características, aprendendo padrões mais abstratos.
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Camada Flatten: transforma a matriz 2D de características em um vetor 1D,
    # preparando os dados para as camadas densas (totalmente conectadas).
    Flatten(),
    
    # Primeira camada densa com 128 neurônios. Atua como um classificador
    # baseado nas características extraídas pelas camadas convolucionais.
    Dense(128, activation='relu'),
    
    # Camada de Dropout: técnica de regularização que zera aleatoriamente 50%
    # das entradas durante o treinamento para prevenir sobreajuste (overfitting).
    Dropout(0.5),

    # Camada de saída: possui um neurônio para cada classe. A função de ativação
    # 'softmax' converte as saídas em uma distribuição de probabilidade,
    # indicando a confiança do modelo para cada classe.
    Dense(y_encoded.shape[1], activation='softmax')
])

# --- Etapa 3: Compilação e Treinamento do Modelo ---

# Compila o modelo, definindo seus componentes para o processo de treinamento.
# 'optimizer': Adam, um algoritmo de otimização eficiente.
# 'loss': 'categorical_crossentropy', função de perda padrão para classificação multiclasse.
# 'metrics': 'accuracy', a métrica a ser monitorada durante o treinamento.
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Exibe um resumo da arquitetura do modelo no console.
model.summary()

# Define um 'callback' para salvar o modelo. O ModelCheckpoint monitora a
# acurácia de validação ('val_accuracy') e salva no disco apenas o modelo
# que apresentar o melhor desempenho até o momento ('save_best_only=True').
checkpoint = ModelCheckpoint(
    "models/best_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1
)

print("[INFO] Iniciando o treinamento do modelo...")

# Executa o treinamento do modelo.
model.fit(
    X_train, y_train,              # Dados de treinamento.
    epochs=10,                     # Número de vezes que o modelo verá o dataset completo.
    batch_size=32,                 # Número de amostras por atualização de gradiente.
    validation_data=(X_val, y_val),# Dados para validação ao final de cada época.
    callbacks=[checkpoint]         # Lista de callbacks a serem ativados.
)

print("[INFO] Treinamento finalizado. Melhor modelo salvo em 'models/best_model.keras'.")