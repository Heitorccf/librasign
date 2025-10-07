# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import kagglehub

def normalize_landmarks(data):
    """
    Normalizando um conjunto de landmarks para ser invariante
    à posição e à escala da mão.
    """
    normalized_data = []
    # Iterando sobre cada amostra de dados.
    for sample in data:
        # Remodelando para 21 pontos (x, y, z).
        landmarks = sample.reshape((21, 3))

        # Centralizando os pontos em relação ao pulso.
        base_point = landmarks[0].copy()
        relative_landmarks = landmarks - base_point

        # Normalizando pela distância entre o pulso e a base do dedo médio.
        scale_dist = np.linalg.norm(relative_landmarks[9])
        if scale_dist < 1e-6: # Evitando divisão por zero.
            scale_dist = 1.0

        scaled_landmarks = relative_landmarks / scale_dist
        # Achatando de volta para 63 coordenadas.
        normalized_data.append(scaled_landmarks.flatten())
    return np.array(normalized_data)

print("[INFO] Carregando dataset de landmarks")

# Baixando o dataset do Kaggle Hub para garantir consistência.
print("[INFO] Baixando o dataset 'librasign' do Kaggle Hub")
kaggle_dataset_path = kagglehub.dataset_download("heitorccf/librasign")

# Ajustando o DATA_DIR para a subpasta correta dos CSVs.
DATA_DIR = os.path.join(kaggle_dataset_path, "landmarks")

print(f"[INFO] Lendo dados de: {DATA_DIR}")

# Inicializando listas para características (X) e rótulos (y).
X_raw, y_raw = [], []

# Iterando sobre os arquivos CSV para construir a matriz de características.
for file in os.listdir(DATA_DIR):
    if file.endswith('.csv'):
        # Extraindo o rótulo do nome do arquivo.
        label = file.split('.')[0]
        df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
        # Adicionando os valores dos landmarks.
        X_raw.append(df.values)
        # Duplicando o rótulo para cada amostra no arquivo.
        y_raw.extend([label] * len(df))

# Verificando se algum dado foi carregado.
if not X_raw:
    raise ValueError(f"Nenhum arquivo .csv encontrado no diretório {DATA_DIR}")

# Empilhando os arrays de características.
X = np.vstack(X_raw)

# Aplicando a codificação numérica aos rótulos textuais.
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Persistindo o mapeamento de classes para uso na inferência.
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
np.save(os.path.join(MODELS_DIR, 'classes.npy'), le.classes_)

print("[INFO] Normalizando landmarks para invariância de posição e escala")
X_normalized = normalize_landmarks(X)

print("[INFO] Iniciando treinamento com validação cruzada (K=5)")
n_splits = 5
# Usando StratifiedKFold para manter a proporção das classes nos folds.
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
accuracies = []
all_y_true = []
all_y_pred = []

# Iterando sobre cada fold da validação cruzada.
for i, (train_index, test_index) in enumerate(skf.split(X_normalized, y)):
    print(f"--- FOLD {i + 1}/{n_splits} ---")
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Aplicando StandardScaler aos dados de cada fold.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Inicializando e treinando o classificador MLP.
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64), # Arquitetura das camadas ocultas.
        max_iter=300,                  # Máximo de iterações do otimizador.
        random_state=42,               # Semente para reprodutibilidade.
        verbose=False,                 # Desativando a saída detalhada do treinamento.
        alpha=0.001                    # Parâmetro de regularização L2.
    )
    # Treinando o modelo no conjunto de treinamento.
    model.fit(X_train_scaled, y_train)

    # Realizando predições no conjunto de teste.
    y_pred = model.predict(X_test_scaled)
    # Calculando e armazenando a acurácia.
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Acurácia do Fold {i + 1}: {accuracy * 100:.2f}%")

    # Coletando rótulos e predições para a matriz de confusão final.
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# Calculando a acurácia média e o desvio padrão.
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print("-" * 30)
print(f"[RESULTADO] Acurácia Média: {mean_accuracy * 100:.2f}% (+/- {std_accuracy * 100:.2f}%)")

# Gerando e salvando a matriz de confusão.
conf_matrix = confusion_matrix(all_y_true, all_y_pred)
np.save(os.path.join(MODELS_DIR, 'confusion_matrix.npy'), conf_matrix)
print(f"[INFO] Matriz de confusão salva em '{os.path.join(MODELS_DIR, 'confusion_matrix.npy')}'")

print("[INFO] Treinando o modelo final com todo o dataset")
# Treinando o modelo uma última vez com todos os dados para maximizar o aprendizado.
final_scaler = StandardScaler().fit(X_normalized)
X_final_scaled = final_scaler.transform(X_normalized)
final_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=300,
    random_state=42,
    alpha=0.001
)
final_model.fit(X_final_scaled, y)

print("[INFO] Salvando artefatos finais do modelo")
# Persistindo o modelo treinado e o scaler para uso em tempo real.
with open(os.path.join(MODELS_DIR, "librasign_mlp.pkl"), 'wb') as f:
    pickle.dump(final_model, f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), 'wb') as f:
    pickle.dump(final_scaler, f)

print("[INFO] Processo de treinamento aprimorado concluído com sucesso")