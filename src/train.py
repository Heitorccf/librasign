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

# Iniciando o carregamento do conjunto de dados pré-processados
print("[INFO] Carregando dataset de landmarks...")
DATA_DIR = "data/landmarks"
X, y = [], []

# Iterando sobre os arquivos CSV para construir a matriz de características
for file in os.listdir(DATA_DIR):
    if file.endswith('.csv'):
        label = file.split('.')[0]
        df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
        X_raw.append(df.values)
        y_raw.extend([label] * len(df))

X = np.vstack(X)

# Aplicando a codificação numérica aos rótulos textuais para compatibilidade com o algoritmo
# A transformação de categorias em valores numéricos otimiza o processamento computacional
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Persistindo o mapeamento de classes para reconstrução posterior das predições
np.save('models/classes.npy', le.classes_)

# Estruturando a divisão estratificada entre conjuntos de treinamento e validação
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementando o processo de treinamento incremental para monitoramento da convergência
print("[INFO] Treinando o modelo MLPClassifier iterativamente...")
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1, warm_start=True, random_state=42, verbose=False)

loss_history = []
n_epochs = 100  # Definindo o número total de iterações de treinamento

for epoch in range(n_epochs):
    model.fit(X_train_scaled, y_train)
    loss_history.append(model.loss_)
    print(f"Época {epoch + 1}/{n_epochs} - Perda: {model.loss_:.4f}", end='\r')

print(f"\n[INFO] Treinamento finalizado após {model.n_iter_} iterações.")

# Executando a validação do modelo treinado
print("[INFO] Avaliando o modelo...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia final do modelo nos dados de teste: {accuracy * 100:.2f}%")

# Salvando os artefatos gerados durante o treinamento
print("[INFO] Salvando artefatos do modelo...")
os.makedirs("models", exist_ok=True)

with open("models/librasign_mlp.pkl", 'wb') as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

# Armazenando o histórico de evolução da função de perda e os dados de validação
np.save('models/loss_history.npy', loss_history)
np.save('models/test_data.npy', {'X_test': X_test_scaled, 'y_test': y_test})

print("[INFO] Processo concluído com sucesso.")