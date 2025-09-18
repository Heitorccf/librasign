# -*- coding: utf-8 -*-
"""
Sistema de Carregamento e Pré-processamento de Dados Visuais (Legado)

Aviso: Este módulo está representando uma abordagem de processamento baseada em imagens
e não está sendo utilizado pela metodologia atual do projeto, que opera
diretamente sobre coordenadas geométricas (landmarks)

Este arquivo está sendo mantido para fins de documentação e para eventuais
desenvolvedores que desejem explorar uma alternativa de pipeline utilizando
redes neurais convolucionais, que operam sobre dados de imagem
"""
import os
import cv2
import numpy as np

# ============================================================================
# Bloco de Configuração
# ============================================================================

# Especificando o diretório fonte para as imagens brutas
# A estrutura esperada seria uma pasta para cada classe, contendo as imagens
DATA_DIR = "data/raw"

# Definindo uma dimensão padronizada para as imagens
# O motivo desta padronização é garantir que a entrada da rede neural
# tenha sempre um tamanho uniforme, um requisito para o processamento em lote
IMG_SIZE = 224

# ============================================================================
# Bloco Funcional
# ============================================================================

def load_and_preprocess_images():
    """
    Executa o carregamento e o pré-processamento de um conjunto de dados visual
    
    Este procedimento estaria percorrendo sistematicamente a estrutura de diretórios,
    processando as imagens em escala monocromática, redimensionando-as e normalizando
    seus valores de pixel, visando a otimização de um eventual processo de treinamento
    
    Returns:
        tuple: Uma estrutura contendo dois arrays NumPy:
               - data: Um tensor multidimensional com as imagens processadas
               - labels: Um vetor com os identificadores categóricos correspondentes
    """
    data = []
    labels = []
    
    # Percorrendo a estrutura de diretórios para identificar as classes
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        
        # Validando se o item é um diretório, ignorando outros arquivos
        if not os.path.isdir(label_path):
            continue
        
        # Processando individualmente cada arquivo de imagem da classe atual
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            
            # Carregando a imagem diretamente em escala de cinza
            # A decisão por esta abordagem visa otimizar o consumo de memória
            # e reduzir a complexidade do modelo, focando apenas na forma
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Verificando se a imagem foi carregada corretamente
            if img is None:
                print(f"[AVISO] Não foi possível ler a imagem: {img_path}")
                continue
            
            # Aplicando o redimensionamento para garantir uniformidade dimensional
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Normalizando os valores de pixel para o intervalo [0, 1]
            # Esta transformação para ponto flutuante ajuda a estabilizar
            # e acelerar a convergência durante o treinamento do modelo
            img = img.astype(np.float32) / 255.0
            
            # Acumulando a imagem processada e seu rótulo correspondente
            data.append(img)
            labels.append(label)
    
    # Convertendo as listas para arrays NumPy
    # O objetivo é obter uma estrutura de dados otimizada para
    # manipulação numérica e compatível com frameworks de machine learning
    data = np.array(data)
    labels = np.array(labels)
    
    # Expandindo a dimensionalidade do tensor de dados
    # A razão para esta etapa é adequar o formato do array (N, H, W) para
    # (N, H, W, C), o formato esperado por muitas arquiteturas de redes
    # convolucionais, onde C é o número de canais (1 para escala de cinza)
    data = np.expand_dims(data, axis=-1)
    
    return data, labels

# ============================================================================
# Bloco de Validação
# ============================================================================

# Verificando se o script está sendo executado diretamente
# Este bloco serve como uma rotina de teste rápido para validar
# a funcionalidade do pipeline de carregamento e processamento
if __name__ == "__main__":
    X, y = load_and_preprocess_images()
    print(f"Total de imagens carregadas: {len(X)}")
    print(f"Formato dos dados das imagens (com canal): {X.shape}")
    
    # Utilizando uma operação de conjunto para identificar as classes únicas
    # que foram encontradas e processadas no diretório de dados
    print(f"Classes (rótulos) encontradas: {sorted(list(set(y)))}")