# -*- coding: utf-8 -*-
"""
Sistema de Aquisição de Dados Geométricos para Reconhecimento Gestual

Este módulo está implementando a captura sistemática de coordenadas da mão,
persistindo os dados brutos em formato CSV para posterior processamento
A arquitetura está focando na estabilidade operacional e na clareza do
feedback visual durante a aquisição
"""
import cv2
import mediapipe as mp
import os
import numpy as np
import csv

# ============================================================================
# Bloco de Configuração e Inicialização
# ============================================================================

# Definindo o diretório de saída para os dados,
# organizando os datasets em uma pasta dedicada
DATA_DIR = "data/landmarks"
CAPTURE_LIMIT = 1000  # Estabelecendo um limite de amostras por classe para evitar desbalanceamento

# Inicializando a solução MediaPipe Hands com parâmetros específicos
# A decisão por estes valores de confiança visa equilibrar performance e precisão,
# evitando detecções falsas e mantendo um rastreamento estável
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils # Obtendo o utilitário de desenho para visualização

# Inicializando a captura de vídeo a partir da webcam padrão (índice 0)
cap = cv2.VideoCapture(0)

# Verificando se a conexão com a câmera foi bem-sucedida
# Esta validação previne a execução do loop principal caso a câmera não esteja disponível
if not cap.isOpened():
    print("[ERRO] Não foi possível abrir a câmera, verifique a conexão e as permissões")
    exit()

# Criando a janela de visualização explicitamente antes do loop
# O motivo desta abordagem é garantir um comportamento consistente da GUI,
# prevenindo a criação de múltiplas janelas em alguns gerenciadores gráficos
WINDOW_NAME = "Captura de Landmarks - LIBRAS"
cv2.namedWindow(WINDOW_NAME)

# Definindo variáveis de estado para gerenciar o fluxo de captura
is_capturing = False
current_label = None
capture_count = 0

# Exibindo as instruções de uso para o operador no console
print("-" * 50)
print("Ferramenta de Captura de Landmarks - LIBRAS")
print("INSTRUÇÕES:")
print(" > Pressione uma letra (A-Z) ou 0 para iniciar a captura")
print(" > Pressione 'ESPACO' para pausar ou retomar")
print(" > Pressione 'ESC' para sair")
print("-" * 50)

# ============================================================================
# Bloco do Loop Principal de Execução
# ============================================================================

# Iniciando o loop para processamento contínuo dos frames da câmera
while True:
    ret, frame = cap.read() # Lendo um novo frame
    if not ret:
        print("[AVISO] Frame da câmera não pôde ser lido, encerrando")
        break

    # Espelhando o frame horizontalmente
    # Esta transformação cria um efeito de espelho, melhorando a usabilidade
    # e a intuição do usuário ao se posicionar
    frame = cv2.flip(frame, 1)

    # Convertendo o frame de BGR para RGB, o formato esperado pelo MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Otimizando o desempenho ao passar o frame como não gravável,
    # permitindo que o MediaPipe processe a imagem por referência
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    # Verificando se landmarks de mão foram detectados
    if results.multi_hand_landmarks:
        # Acessando o primeiro (e único) conjunto de landmarks de mão detectado
        hand_landmarks = results.multi_hand_landmarks[0]

        # Renderizando os landmarks e as conexões sobre o frame
        # Este feedback visual é crucial para o usuário posicionar a mão corretamente
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Verificando se a captura está ativa e dentro do limite estabelecido
        if is_capturing and current_label and capture_count < CAPTURE_LIMIT:
            try:
                # Extraindo as coordenadas brutas (x, y, z) dos 21 landmarks
                # e achatando o array para um vetor de 63 dimensões
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                # Garantindo que o diretório de dados exista antes de salvar
                os.makedirs(DATA_DIR, exist_ok=True)
                csv_path = os.path.join(DATA_DIR, f"{current_label}.csv")

                # Abrindo o arquivo CSV no modo 'append' ('a')
                # A escolha por este modo permite adicionar novas amostras
                # sem apagar os dados capturados anteriormente
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(coords) # Escrevendo o vetor de coordenadas

                capture_count += 1 # Incrementando o contador de amostras da classe atual
            except Exception as e:
                print(f"[ERRO] Falha ao salvar os dados: {e}")

    # ============================================================================
    # Bloco de Interface e Feedback Visual
    # ============================================================================

    # Gerenciando dinamicamente o texto de status para feedback ao usuário
    if current_label:
        if capture_count >= CAPTURE_LIMIT:
            is_capturing = False
            status_text = f"CLASSE '{current_label}' COMPLETA ({capture_count}/{CAPTURE_LIMIT})"
        elif is_capturing:
            status_text = f"GRAVANDO '{current_label}' ({capture_count}/{CAPTURE_LIMIT})"
        else:
            status_text = f"PAUSADO '{current_label}' ({capture_count}/{CAPTURE_LIMIT})"
    else:
        status_text = "Ocioso, escolha uma letra para iniciar"

    # Desenhando o texto de status sobre o frame
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow(WINDOW_NAME, frame) # Atualizando a janela com o frame processado

    # ============================================================================
    # Bloco de Controle via Teclado
    # ============================================================================

    # Aguardando por uma tecla pressionada e aplicando uma máscara para compatibilidade
    key = cv2.waitKey(5) & 0xFF

    if key == 27: # 27 é o código ASCII para a tecla ESC
        break
    elif key == 32: # 32 é o código ASCII para a barra de espaço
        if current_label:
            is_capturing = not is_capturing # Alternando o estado de captura
    # Verificando se a tecla está no intervalo de letras (maiúsculas ou minúsculas) ou é '0'
    elif (ord('a') <= key <= ord('z')) or (ord('A') <= key <= ord('Z')) or (key == ord('0')):
        label_name = "nenhum" if key == ord('0') else chr(key).upper()
        current_label = label_name
        csv_path = os.path.join(DATA_DIR, f"{current_label}.csv")

        # Verificando a contagem de amostras existentes para a classe selecionada
        # A decisão de ler o arquivo ao invés de manter um contador em memória
        # torna o processo mais robusto, permitindo continuar capturas interrompidas
        try:
            with open(csv_path, 'r') as f:
                capture_count = sum(1 for _ in f)
        except FileNotFoundError:
            capture_count = 0

        if capture_count < CAPTURE_LIMIT:
            is_capturing = True
            print(f"\n[INFO] Iniciando captura para '{current_label}', capturas existentes: {capture_count}")
        else:
            is_capturing = False
            print(f"\n[AVISO] Classe '{current_label}' já atingiu o limite de {CAPTURE_LIMIT} amostras")

# ============================================================================
# Bloco de Finalização
# ============================================================================

# Liberando os recursos utilizados de forma segura ao final da execução
print("[INFO] Encerrando aplicação")
hands.close()
cap.release()
cv2.destroyAllWindows()