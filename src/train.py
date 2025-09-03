# -*- coding: utf-8 -*-
"""
Sistema de Treinamento de Rede Neural com Validação Cruzada e Normalização de Landmarks.

Este módulo implementa um pipeline de treinamento robusto que inclui:
1. Normalização de coordenadas geométricas para garantir invariância à posição e escala.
2. Treinamento de um classificador MLP usando validação cruzada estratificada para uma
   avaliação de desempenho mais confiável.
3. Geração e salvamento de artefatos, incluindo o modelo treinado, o normalizador,
   o mapeamento de classes e a matriz de confusão para análise de erros.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# --- 1. FUNÇÃO DE NORMALIZAÇÃO DE LANDMARKS ---
def normalize_landmarks(data):
    """
    Normaliza um conjunto de landmarks para ser invariante à posição e escala.

    Args:
        data (np.array): Array de formato (n_samples, 63) com as coordenadas.

    Returns:
        np.array: Array de formato (n_samples, 63) com os dados normalizados.
    """
    normalized_data = []
    for sample in data:
        # Remodelar a amostra para (21, 3) para facilitar os cálculos
        landmarks = sample.reshape((21, 3))

        # Passo 1: Invariância à Posição (Translação)
        # Centralizar todos os pontos em relação ao pulso (landmark 0)
        base_point = landmarks[0].copy()
        relative_landmarks = landmarks - base_point

        # Passo 2: Invariância à Escala
        # Calcular a distância entre o pulso e a base do dedo médio (landmark 9)
        # Se essa distância for zero, use um valor pequeno para evitar divisão por zero
        scale_dist = np.linalg.norm(relative_landmarks[9])
        if scale_dist < 1e-6:
            scale_dist = 1.0 # Evita divisão por zero em casos anômalos

        # Normalizar pela distância
        scaled_landmarks = relative_landmarks / scale_dist

        # Aplainar de volta para o formato (63,) e adicionar à lista
        normalized_data.append(scaled_landmarks.flatten())

    return np.array(normalized_data)

# --- 2. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ---
print("[INFO] Carregando dataset de landmarks...")
DATA_DIR = "data/landmarks"
X_raw, y_raw = [], []

# Iterando sobre os arquivos CSV para construir a matriz de características
for file in os.listdir(DATA_DIR):
    if file.endswith('.csv'):
        label = file.split('.')[0]
        df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
        X_raw.append(df.values)
        y_raw.extend([label] * len(df))

X = np.vstack(X_raw)

# Aplicando a codificação numérica aos rótulos
le = LabelEncoder()
y = le.fit_transform(y_raw)
np.save('models/classes.npy', le.classes_) # Salva as classes para uso na predição

# --- 3. APLICANDO A NORMALIZAÇÃO DE LANDMARKS ---
print("[INFO] Normalizando landmarks para invariância de posição e escala...")
X_normalized = normalize_landmarks(X)

# --- 4. TREINAMENTO COM VALIDAÇÃO CRUZADA E ANÁLISE ---
print("[INFO] Iniciando treinamento com validação cruzada (K=5)...")

# Configuração da Validação Cruzada Estratificada
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Listas para armazenar métricas de cada fold
accuracies = []
all_y_true = []
all_y_pred = []

# Loop de treinamento
for i, (train_index, test_index) in enumerate(skf.split(X_normalized, y)):
    print(f"--- FOLD {i + 1}/{n_splits} ---")
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Normalização dos dados (StandardScaler) - ajustado a cada fold de treino
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definição do modelo com regularização L2 (alpha)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300, # Aumentar para garantir convergência
        random_state=42,
        verbose=False,
        alpha=0.001 # Parâmetro de regularização L2
    )

    # Treinamento
    model.fit(X_train_scaled, y_train)

    # Avaliação
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Acurácia do Fold {i + 1}: {accuracy * 100:.2f}%")

    # Guardar resultados para a matriz de confusão final
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# --- 5. RESULTADOS FINAIS E SALVAMENTO DOS ARTEFATOS ---
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print("-" * 30)
print(f"[RESULTADO] Acurácia Média: {mean_accuracy * 100:.2f}% (+/- {std_accuracy * 100:.2f}%)")

# Geração da matriz de confusão
conf_matrix = confusion_matrix(all_y_true, all_y_pred)
np.save('models/confusion_matrix.npy', conf_matrix)
print("[INFO] Matriz de confusão salva em 'models/confusion_matrix.npy'")

# Treinando o modelo final com TODOS os dados para produção
print("[INFO] Treinando o modelo final com todo o dataset...")
final_scaler = StandardScaler().fit(X_normalized)
X_final_scaled = final_scaler.transform(X_normalized)

final_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=300,
    random_state=42,
    alpha=0.001
)
final_model.fit(X_final_scaled, y)

# Salvando os artefatos finais
print("[INFO] Salvando artefatos finais do modelo...")
os.makedirs("models", exist_ok=True)
with open("models/librasign_mlp.pkl", 'wb') as f:
    pickle.dump(final_model, f)
with open("models/scaler.pkl", 'wb') as f:
    pickle.dump(final_scaler, f)

print("[INFO] Processo de treinamento aprimorado concluído com sucesso.")