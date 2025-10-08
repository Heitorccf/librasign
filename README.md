# LibraSign

## English

### Introduction

This repository contains the source code for the LibraSign project, a real-time Brazilian Sign Language (Libras) translator. For a comprehensive understanding of the project's theoretical foundation, methodology, and results, please consult the complete academic thesis, available in this repository:

  - **[HeitorFernandes-TCC\_BSI.pdf](https://github.com/Heitorccf/librasign/blob/master/HeitorFernandes-TCC_BSI.pdf)**

-----

### Project Overview

The primary objective of this project is to develop an open-source software that utilizes computer vision and machine learning techniques to translate gestures from the Brazilian Sign Language (Libras) into text in real-time. The initiative is motivated by the need to promote communication accessibility and social inclusion for the deaf community. The current scope of the system is exclusively focused on translating the manual alphabet of Libras (letters A-Z), providing a foundational framework for future expansion.

-----

### Dataset

The model was trained using a custom dataset of geometric hand landmarks, captured specifically for this project. The dataset is publicly available on Kaggle and can be accessed via the following link:

  - **[Libras Landmark Dataset (A-Z)](https://www.kaggle.com/datasets/heitorccf/librasign)**

-----

### Functionality

The system operates through a modular pipeline that consists of data capture, model training, and real-time prediction.

#### **Execution Flow**

1.  **Data Capture (`src/capture.py`):** This script uses a webcam to capture hand gestures. The **MediaPipe** library detects and extracts the 3D coordinates of 21 key hand landmarks for each gesture. This geometric data, rather than raw image pixels, is saved into `.csv` files, creating a lightweight and efficient dataset.
2.  **Model Training (`src/train.py`):** This script processes the landmark data collected in the previous step. It applies normalization to ensure the data is invariant to hand position and scale. A **Multi-layer Perceptron (MLP)** classifier is then trained on this structured data. The trained model, a data scaler, and class mappings are saved as artifacts for the prediction phase.
3.  **Real-time Prediction (`src/predict.py`):** This is the main application script. It loads the trained MLP model and uses the webcam to capture gestures in real-time. For each frame, it extracts hand landmarks, applies the same normalization and scaling transformations used during training, and feeds the data to the model for classification. The predicted letter is then displayed on the screen.

#### **Usage Instructions**

It is highly recommended to use a virtual development environment to manage dependencies and avoid conflicts.

1.  **Clone the repository and set up a virtual environment:**

    ```bash
    git clone https://github.com/heitorccf/librasign.git
    cd librasign

    # Create the virtual environment
    python -m venv .venv # Linux and MacOS

    # Activate the virtual environment
    source .venv/bin/activate # Linux and MacOS
    .venv\Scripts\activate # Windows
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Execution Order:** The scripts must be executed in the following order:

      * **To capture your own dataset (optional):**
        ```bash
        python src/capture.py
        ```
      * **To train the model on the dataset:**
        ```bash
        python src/train.py
        ```
      * **To run the real-time translator:**
        ```bash
        python src/predict.py
        ```

-----

### Requirements

  - **Recommended Python Version:** `3.11.13`

  - **Main Frameworks for Machine Learning and Data Processing:**

      - `scikit-learn==1.7.2`
      - `numpy==2.2.6`
      - `pandas==2.3.2`

  - **Libraries for Image Processing and Computer Vision:**

      - `opencv-python==4.12.0.88`
      - `mediapipe==0.10.14`

  - **Tool for Downloading Kaggle Datasets:**

      - `kagglehub==0.3.13`

-----

### Applicability and Scalability

Although this project was developed for the Brazilian Sign Language alphabet, its architecture is highly adaptable. The system can be retrained to recognize gestures from other sign languages or even custom gesture sets. To do so, a new dataset must be collected using the `capture.py` script, and the model must be retrained with the new data using the `train.py` script. This modularity highlights the project's flexibility and potential for scalability.

-----

### Further Information

For any questions or a deeper understanding of the project, including the literature review, architectural decisions, and detailed analysis, readers are strongly encouraged to consult the referenced **.pdf document**.

-----

-----

## Português (Brasil)

### Introdução

Este repositório contém o código-fonte do projeto LibraSign, um tradutor de Língua Brasileira de Sinais (Libras) em tempo real. Para uma compreensão abrangente da fundamentação teórica, metodologia e resultados do projeto, por favor, consulte o trabalho de conclusão de curso completo, disponível neste repositório:

  - **[HeitorFernandes-TCC\_BSI.pdf](https://github.com/Heitorccf/librasign/blob/master/HeitorFernandes-TCC_BSI.pdf)**

-----

### Visão Geral do Projeto

O objetivo principal deste projeto é o desenvolvimento de um software de código aberto que utiliza técnicas de visão computacional e aprendizado de máquina para traduzir, em tempo real, os gestos da Língua Brasileira de Sinais (Libras) em texto. A iniciativa é motivada pela necessidade de promover a acessibilidade comunicacional e a inclusão social da comunidade surda. O escopo atual do sistema está exclusivamente focado na tradução do alfabeto manual da Libras (letras de A a Z), fornecendo uma estrutura fundamental para futuras expansões.

-----

### Conjunto de Dados

O modelo foi treinado com um conjunto de dados customizado de marcos geométricos de mãos, capturado especificamente para este projeto. O dataset está publicamente disponível no Kaggle e pode ser acessado através do seguinte link:

  - **[Libras Landmark Dataset (A-Z)](https://www.kaggle.com/datasets/heitorccf/librasign)**

-----

### Funcionalidade

O sistema opera através de um pipeline modular que consiste em captura de dados, treinamento do modelo e predição em tempo real.

#### **Fluxo de Execução**

1.  **Captura de Dados (`src/capture.py`):** Este script utiliza uma webcam para capturar gestos manuais. A biblioteca **MediaPipe** detecta e extrai as coordenadas 3D de 21 pontos de referência (landmarks) da mão para cada gesto. Esses dados geométricos, em vez de pixels de imagem, são salvos em arquivos `.csv`, criando um dataset leve e eficiente.
2.  **Treinamento do Modelo (`src/train.py`):** Este script processa os dados de landmarks coletados na etapa anterior. Ele aplica uma normalização para garantir que os dados sejam invariantes à posição e à escala da mão. Em seguida, um classificador **Multi-layer Perceptron (MLP)** é treinado com esses dados estruturados. O modelo treinado, um normalizador de dados (*scaler*) e os mapeamentos de classe são salvos como artefatos para a fase de predição.
3.  **Predição em Tempo Real (`src/predict.py`):** Este é o script principal da aplicação. Ele carrega o modelo MLP treinado e utiliza a webcam para capturar gestos em tempo real. Para cada quadro de vídeo, ele extrai os landmarks da mão, aplica as mesmas transformações de normalização e escala usadas no treinamento, e submete os dados ao modelo para classificação. A letra predita é então exibida na tela.

#### **Instruções de Uso**

É altamente recomendável o uso de um ambiente virtual de desenvolvimento para gerenciar as dependências e evitar conflitos.

1.  **Clone o repositório e configure um ambiente virtual:**

    ```bash
    git clone https://github.com/heitorccf/librasign.git
    cd librasign

    # Crie o ambiente virtual
    python -m venv .venv # Linux e MacOS

    # Ative o ambiente virtual
    source .venv/bin/activate # Linux e MacOS
    .venv\Scripts\activate # Windows
    ```

2.  **Instale as dependências necessárias:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Ordem de Execução:** Os scripts devem ser executados na seguinte ordem:

      * **Para capturar seu próprio dataset (opcional):**
        ```bash
        python src/capture.py
        ```
      * **Para treinar o modelo com o dataset:**
        ```bash
        python src/train.py
        ```
      * **Para executar o tradutor em tempo real:**
        ```bash
        python src/predict.py
        ```

-----

### Requisitos

  - **Versão do Python recomendada:** `3.11.13`

  - **Frameworks principais de Machine Learning e processamento de dados:**

      - `scikit-learn==1.7.2`
      - `numpy==2.2.6`
      - `pandas==2.3.2`

  - **Bibliotecas para processamento de imagem e visão computacional:**

      - `opencv-python==4.12.0.88`
      - `mediapipe==0.10.14`

  - **Ferramenta para download de datasets do Kaggle:**

      - `kagglehub==0.3.13`

-----

### Aplicabilidade e Escalabilidade

Embora este projeto tenha sido desenvolvido para o alfabeto da Língua Brasileira de Sinais, sua arquitetura é altamente adaptável. O sistema pode ser retreinado para reconhecer gestos de outras línguas de sinais ou até mesmo conjuntos de gestos personalizados. Para isso, um novo conjunto de dados deve ser coletado utilizando o script `capture.py`, e o modelo deve ser retreinado com os novos dados por meio do script `train.py`. Essa modularidade evidencia a flexibilidade e o potencial de escalabilidade do projeto.

-----

### Informações Adicionais

Para quaisquer dúvidas ou para um aprofundamento no projeto, incluindo a revisão de literatura, as decisões de arquitetura e análises detalhadas, recomenda-se fortemente que os leitores consultem o **documento .pdf** referenciado.