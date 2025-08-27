# -*- coding: utf-8 -*-
"""
Treinamento de um Modelo Leve (MLP) com Dados de Landmarks e
Geração de Histórico para Análise.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# --- Carregamento dos Dados ---
print("[INFO] Carregando dataset de landmarks...")
DATA_DIR = "data/landmarks"
X, y = [], []

# Carrega os dados e os rótulos de cada arquivo .csv
for file in os.listdir(DATA_DIR):
    if file.endswith('.csv'):
        label = file.split('.')[0]
        df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
        X.append(df.values)
        y.extend([label] * len(df))

X = np.vstack(X)

# --- MUDANÇA: Usando LabelEncoder para converter rótulos de texto para números ---
# O scikit-learn funciona melhor com rótulos numéricos.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Salva as classes para uso posterior no notebook
np.save('models/classes.npy', le.classes_)

# --- Preparação dos Dados ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MUDANÇA: Treinamento Iterativo para Gerar a Curva de Perda ---
print("[INFO] Treinando o modelo MLPClassifier iterativamente...")
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1, warm_start=True, random_state=42, verbose=False)

loss_history = []
n_epochs = 100 # Número de "épocas" que vamos treinar

for epoch in range(n_epochs):
    model.fit(X_train_scaled, y_train)
    loss_history.append(model.loss_)
    print(f"Época {epoch + 1}/{n_epochs} - Perda: {model.loss_:.4f}", end='\r')

print(f"\n[INFO] Treinamento finalizado após {model.n_iter_} iterações.")

# --- Avaliação ---
print("[INFO] Avaliando o modelo...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia final do modelo nos dados de teste: {accuracy * 100:.2f}%")

# --- Salvando o Modelo, o Scaler e o Histórico ---
print("[INFO] Salvando artefatos do modelo...")
os.makedirs("models", exist_ok=True)
with open("models/librasign_mlp.pkl", 'wb') as f:
    pickle.dump(model, f)
with open("models/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

# Salva o histórico de perda e os dados de teste para o notebook
np.save('models/loss_history.npy', loss_history)
np.save('models/test_data.npy', {'X_test': X_test_scaled, 'y_test': y_test})

print("[INFO] Processo concluído com sucesso.")