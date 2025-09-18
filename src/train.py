# -*- coding: utf-8 -*-
"""
Sistema de Treinamento de Rede Neural com Validação Cruzada e Normalização de Landmarks

Este módulo está implementando um pipeline de treinamento robusto que inclui:
1, Normalização de coordenadas geométricas para garantir invariância à posição e escala
2, Treinamento de um classificador MLP utilizando validação cruzada estratificada para uma
   avaliação de desempenho mais confiável e generalizável
3, Geração e salvamento de artefatos essenciais, como o modelo treinado, o normalizador
   de recursos, o mapeamento de classes e a matriz de confusão para análise de erros
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import kagglehub # Reintroduzindo kagglehub para carregar o dataset

# ============================================================================
# Bloco de Funções Auxiliares
# ============================================================================

def normalize_landmarks(data):
    """
    Normaliza um conjunto de landmarks para ser invariante à posição e escala

    Esta função está processando cada amostra, ajustando os landmarks
    para que o pulso fique na origem (invariância à posição) e dimensionando
    todo o conjunto de pontos com base na distância entre o pulso e a base
    do dedo médio (invariância à escala)
    """
    normalized_data = []
    # Iterando sobre cada amostra de dados de landmarks
    for sample in data:
        landmarks = sample.reshape((21, 3)) # Remodelando para 21 pontos (x, y, z)

        # Invariância à Posição: Centrando todos os pontos em relação ao pulso
        base_point = landmarks[0].copy()
        relative_landmarks = landmarks - base_point

        # Invariância à Escala: Normalizando pela distância entre pulso e base do dedo médio
        scale_dist = np.linalg.norm(relative_landmarks[9])
        if scale_dist < 1e-6: # Evitando divisão por zero em casos de degeneração
            scale_dist = 1.0

        scaled_landmarks = relative_landmarks / scale_dist
        normalized_data.append(scaled_landmarks.flatten()) # Achatando de volta para 63 coordenadas
    return np.array(normalized_data)

# ============================================================================
# Bloco de Carregamento e Pré-processamento de Dados
# ============================================================================

print("[INFO] Carregando dataset de landmarks")

# Baixando o dataset do Kaggle Hub para garantir a consistência
# A decisão de usar o Kaggle Hub está assegurando que o ambiente de treinamento
# sempre utilize a mesma versão dos dados
print("[INFO] Baixando o dataset 'librasign' do Kaggle Hub")
kaggle_dataset_path = kagglehub.dataset_download("heitorccf/librasign")

# Ajustando o DATA_DIR para a subpasta correta onde os CSVs estão localizados
# O motivo desta concatenação é que o Kaggle Hub baixa o dataset
# em uma estrutura que inclui uma pasta raiz e subpastas para os dados
DATA_DIR = os.path.join(kaggle_dataset_path, "landmarks")

print(f"[INFO] Lendo dados de: {DATA_DIR}")

# Inicializando listas para armazenar as características (X) e os rótulos (y)
X_raw, y_raw = [], []

# Iterando sobre os arquivos CSV para construir a matriz de características
# Cada arquivo CSV representa uma classe gestual, contendo múltiplas amostras
for file in os.listdir(DATA_DIR):
    if file.endswith('.csv'):
        label = file.split('.')[0] # Extraindo o rótulo da classe a partir do nome do arquivo
        df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
        X_raw.append(df.values) # Adicionando os valores dos landmarks
        y_raw.extend([label] * len(df)) # Duplicando o rótulo para cada amostra no arquivo

# Verificando se algum dado foi carregado
# Esta validação previne erros de concatenação se nenhum arquivo CSV for encontrado
if not X_raw:
    raise ValueError(f"Nenhum arquivo .csv encontrado no diretório {DATA_DIR}")

X = np.vstack(X_raw) # Empilhando todos os arrays de características verticalmente

# Aplicando a codificação numérica aos rótulos textuais
# Esta transformação é necessária pois os algoritmos de Machine Learning
# operam com valores numéricos para as classes
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Persistindo o mapeamento de classes
# A decisão de salvar este objeto permite traduzir as predições numéricas
# de volta para os rótulos textuais originais durante a inferência
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True) # Garantindo que o diretório de modelos exista
np.save(os.path.join(MODELS_DIR, 'classes.npy'), le.classes_)

# ============================================================================
# Bloco de Normalização de Landmarks e Treinamento do Modelo
# ============================================================================

print("[INFO] Normalizando landmarks para invariância de posição e escala")
X_normalized = normalize_landmarks(X)

print("[INFO] Iniciando treinamento com validação cruzada (K=5)")
n_splits = 5 # Definindo o número de folds para a validação cruzada
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # Usando StratifiedKFold para manter a proporção das classes
accuracies = [] # Lista para armazenar as acurácias de cada fold
all_y_true = [] # Lista para armazenar todos os rótulos verdadeiros
all_y_pred = [] # Lista para armazenar todas as predições

# Iterando sobre cada fold da validação cruzada
for i, (train_index, test_index) in enumerate(skf.split(X_normalized, y)):
    print(f"--- FOLD {i + 1}/{n_splits} ---")
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Aplicando StandardScaler aos dados de cada fold
    # A decisão de aplicar o scaler dentro do loop garante que o scaler
    # seja ajustado apenas com os dados de treinamento de cada fold,
    # evitando data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Inicializando e treinando o classificador MLP
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64), # Definindo a arquitetura das camadas ocultas
        max_iter=300, # Número máximo de iterações do otimizador
        random_state=42, # Semente para reprodutibilidade
        verbose=False, # Desativando a saída detalhada do treinamento do MLP
        alpha=0.001 # Parâmetro de regularização L2
    )
    model.fit(X_train_scaled, y_train) # Treinando o modelo no conjunto de treinamento

    y_pred = model.predict(X_test_scaled) # Realizando predições no conjunto de teste
    accuracy = accuracy_score(y_test, y_pred) # Calculando a acurácia
    accuracies.append(accuracy) # Armazenando a acurácia do fold
    print(f"Acurácia do Fold {i + 1}: {accuracy * 100:.2f}%")

    all_y_true.extend(y_test) # Coletando todos os rótulos verdadeiros
    all_y_pred.extend(y_pred) # Coletando todas as predições

# ============================================================================
# Bloco de Avaliação Final e Salvamento de Artefatos
# ============================================================================

mean_accuracy = np.mean(accuracies) # Calculando a acurácia média
std_accuracy = np.std(accuracies) # Calculando o desvio padrão da acurácia
print("-" * 30)
print(f"[RESULTADO] Acurácia Média: {mean_accuracy * 100:.2f}% (+/- {std_accuracy * 100:.2f}%)")

# Gerando e salvando a matriz de confusão
# Este artefato é útil para analisar os tipos de erros que o modelo está cometendo
conf_matrix = confusion_matrix(all_y_true, all_y_pred)
np.save(os.path.join(MODELS_DIR, 'confusion_matrix.npy'), conf_matrix)
print(f"[INFO] Matriz de confusão salva em '{os.path.join(MODELS_DIR, 'confusion_matrix.npy')}'")

print("[INFO] Treinando o modelo final com todo o dataset")
# Treinando o modelo uma última vez com todos os dados
# O motivo é maximizar o aprendizado do modelo antes de salvá-lo para inferência
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
# Persistindo o modelo treinado e o scaler
# Estes arquivos serão carregados pelo script de inferência para uso em tempo real
with open(os.path.join(MODELS_DIR, "librasign_mlp.pkl"), 'wb') as f:
    pickle.dump(final_model, f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), 'wb') as f:
    pickle.dump(final_scaler, f)

print("[INFO] Processo de treinamento aprimorado concluído com sucesso")