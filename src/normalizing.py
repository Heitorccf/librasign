# -*- coding: utf-8 -*-
"""
Sistema de Carregamento e Pré-processamento de Dados Visuais (Legado)

Aviso: Este módulo representa uma abordagem de processamento baseada em imagens
e não está sendo utilizado pela metodologia atual do projeto, que opera
diretamente sobre coordenadas geométricas (landmarks).

Este arquivo está sendo mantido para fins de documentação.
"""
import os
import cv2
import numpy as np

# Diretório fonte para as imagens brutas.
DATA_DIR = "data/raw"

# Dimensão padronizada para as imagens, garantindo entrada uniforme na rede neural.
IMG_SIZE = 224


def load_and_preprocess_images():
    """
    Executando o carregamento e o pré-processamento de um conjunto de dados visual,
    percorrendo a estrutura de diretórios, processando as imagens em escala
    monocromática, redimensionando-as e normalizando seus valores de pixel.
    
    Returns:
        tuple: Uma estrutura contendo dois arrays NumPy:
               - data: Um tensor com as imagens processadas.
               - labels: Um vetor com os identificadores categóricos.
    """
    data = []
    labels = []
    
    # Percorrendo a estrutura de diretórios para identificar as classes.
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        
        # Validando se o item é um diretório.
        if not os.path.isdir(label_path):
            continue
        
        # Processando individualmente cada arquivo de imagem.
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            
            # Carregando a imagem em escala de cinza para otimizar memória e reduzir complexidade.
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Verificando se a imagem foi carregada corretamente.
            if img is None:
                print(f"[AVISO] Não foi possível ler a imagem: {img_path}")
                continue
            
            # Aplicando o redimensionamento para garantir uniformidade dimensional.
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Normalizando os valores de pixel para o intervalo [0, 1].
            img = img.astype(np.float32) / 255.0
            
            # Acumulando a imagem processada e seu rótulo.
            data.append(img)
            labels.append(label)
    
    # Convertendo as listas para arrays NumPy para otimização numérica.
    data = np.array(data)
    labels = np.array(labels)
    
    # Expandindo a dimensionalidade do tensor para o formato (N, H, W, C).
    data = np.expand_dims(data, axis=-1)
    
    return data, labels


# Verificando se o script está sendo executado diretamente.
if __name__ == "__main__":
    X, y = load_and_preprocess_images()
    print(f"Total de imagens carregadas: {len(X)}")
    print(f"Formato dos dados das imagens (com canal): {X.shape}")
    
    # Utilizando uma operação de conjunto para identificar as classes únicas.
    print(f"Classes (rótulos) encontradas: {sorted(list(set(y)))}")