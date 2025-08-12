# -*- coding: utf-8 -*-
"""
Módulo para Carregamento e Pré-processamento de Imagens.

Este script é responsável por carregar as imagens brutas capturadas,
aplicar um conjunto de transformações de normalização e prepará-las
para serem utilizadas como entrada em um modelo de aprendizado de máquina.
O pré-processamento garante a consistência e a adequação dos dados
para o treinamento da Rede Neural Convolucional (CNN).
"""

import os
import cv2
import numpy as np

# --- Configuração de Parâmetros ---

# Define o diretório de onde as imagens capturadas serão lidas.
DATA_DIR = "data/raw"
# Define o tamanho padrão para o qual as imagens serão redimensionadas.
IMG_SIZE = 224

def load_and_preprocess_images():
    """
    Carrega, pré-processa e organiza as imagens e seus respectivos rótulos.

    A função itera sobre os subdiretórios do DATA_DIR, onde cada subdiretório
    representa uma classe (letra). As imagens são lidas em escala de cinza,
    redimensionadas e normalizadas.

    Returns:
        tuple: Uma tupla contendo dois arrays NumPy:
               - data: Array com os dados das imagens processadas.
               - labels: Array com os rótulos correspondentes a cada imagem.
    """
    data = []
    labels = []

    # Itera sobre cada diretório de classe (ex: 'A', 'B', 'C'...).
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)

        # Garante que o caminho é, de fato, um diretório.
        if not os.path.isdir(label_path):
            continue

        # Itera sobre cada arquivo de imagem dentro do diretório da classe.
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            # Lê a imagem diretamente em escala de cinza (1 canal).
            # Esta abordagem é mais eficiente em memória e processamento do que
            # ler em cores e converter posteriormente.
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"[AVISO] Não foi possível ler a imagem: {img_path}")
                continue

            # Redimensiona a imagem para o tamanho padrão (IMG_SIZE x IMG_SIZE).
            # Esta etapa garante a uniformidade dimensional necessária para a entrada da CNN.
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normaliza os valores dos pixels. A conversão dos pixels de um
            # intervalo [0, 255] para [0, 1] é uma prática padrão que melhora
            # a estabilidade e a velocidade da convergência durante o treinamento.
            img = img.astype(np.float32) / 255.0

            # Adiciona a imagem processada e seu rótulo às listas.
            data.append(img)
            labels.append(label)

    # Converte as listas Python em arrays NumPy para manipulação eficiente.
    data = np.array(data)
    labels = np.array(labels)

    # Adiciona uma dimensão extra ao array de dados das imagens.
    # As CNNs do TensorFlow/Keras esperam um formato de entrada de 4D:
    # (número_de_amostras, altura, largura, número_de_canais).
    # Para escala de cinza, o número de canais é 1.
    # O formato passa de (N, 224, 224) para (N, 224, 224, 1).
    data = np.expand_dims(data, axis=-1)

    return data, labels

# --- Bloco de Execução para Teste ---
# Este bloco é executado apenas quando o script é chamado diretamente.
# Serve para uma verificação rápida da funcionalidade do módulo.
if __name__ == "__main__":
    X, y = load_and_preprocess_images()
    print(f"Total de imagens carregadas: {len(X)}")
    print(f"Formato dos dados das imagens (com canal): {X.shape}")
    # A função `set()` é usada para exibir as classes únicas encontradas.
    print(f"Classes (rótulos) encontradas: {sorted(list(set(y)))}")