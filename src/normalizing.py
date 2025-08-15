# -*- coding: utf-8 -*-
"""
Sistema de Carregamento e Pré-processamento de Dados Visuais.

Este módulo implementa o pipeline de preparação de imagens para treinamento
de modelos de aprendizado profundo. O sistema realiza a leitura dos dados
brutos capturados, aplicando transformações padronizadas de normalização e
estruturação dimensional, garantindo a compatibilidade com arquiteturas de
redes neurais convolucionais e otimizando a eficiência do processo de
aprendizagem.
"""

import os
import cv2
import numpy as np

# Estabelecendo parâmetros operacionais do sistema

# Especificando o diretório fonte contendo as imagens brutas organizadas por classe.
DATA_DIR = "data/raw"

# Definindo as dimensões padronizadas para processamento uniforme das imagens.
IMG_SIZE = 224

def load_and_preprocess_images():
    """
    Executa o carregamento completo e o pré-processamento do conjunto de dados visuais.
    
    O procedimento percorre sistematicamente a estrutura hierárquica de diretórios,
    onde cada subpasta corresponde a uma categoria distinta do alfabeto. As imagens
    são processadas em escala monocromática, redimensionadas para dimensões uniformes
    e normalizadas para otimização do processo de treinamento.
    
    Returns:
        tuple: Estrutura contendo dois arrays NumPy organizados:
               - data: Tensor multidimensional contendo as imagens processadas.
               - labels: Vetor de identificadores categóricos correspondentes.
    """
    data = []
    labels = []
    
    # Percorrendo a estrutura de diretórios categorizados por classe alfabética.
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        
        # Validando a natureza diretorial do caminho antes do processamento.
        if not os.path.isdir(label_path):
            continue
        
        # Processando individualmente cada arquivo de imagem da categoria atual.
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            
            # Carregando a imagem diretamente em representação monocromática,
            # otimizando o consumo de memória e eliminando etapas desnecessárias
            # de conversão cromática posterior.
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"[AVISO] Não foi possível ler a imagem: {img_path}")
                continue
            
            # Aplicando redimensionamento para garantir uniformidade dimensional,
            # requisito fundamental para processamento em lote por redes convolucionais.
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Normalizando a distribuição de intensidades luminosas, transformando
            # valores discretos do intervalo [0, 255] para valores contínuos em [0, 1],
            # facilitando a convergência numérica durante o treinamento do modelo.
            img = img.astype(np.float32) / 255.0
            
            # Acumulando a imagem processada e sua classificação correspondente.
            data.append(img)
            labels.append(label)
    
    # Convertendo estruturas de lista para arrays NumPy, proporcionando
    # manipulação vetorizada eficiente e compatibilidade com frameworks de ML.
    data = np.array(data)
    labels = np.array(labels)
    
    # Expandindo a dimensionalidade do tensor de imagens para conformidade
    # com a expectativa arquitetural das redes convolucionais modernas.
    # A transformação adiciona um eixo de canal, convertendo o formato
    # tridimensional (N, altura, largura) para quadridimensional
    # (N, altura, largura, canais), onde N representa o número de amostras
    # e o canal único indica a natureza monocromática dos dados.
    data = np.expand_dims(data, axis=-1)
    
    return data, labels

# Implementando rotina de validação funcional

# Executando verificação diagnóstica quando o módulo é invocado diretamente,
# permitindo validação rápida da integridade do pipeline de processamento.
if __name__ == "__main__":
    X, y = load_and_preprocess_images()
    print(f"Total de imagens carregadas: {len(X)}")
    print(f"Formato dos dados das imagens (com canal): {X.shape}")
    
    # Utilizando operação de conjunto para identificação das categorias
    # únicas presentes no conjunto de dados processado.
    print(f"Classes (rótulos) encontradas: {sorted(list(set(y)))}")