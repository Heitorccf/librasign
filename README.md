# Aplicação de Redes Neurais para Tradução de Libras

**Autor:** Heitor Câmara Costa Fernandes

## Resumo do Projeto

Este projeto apresenta o desenvolvimento de um sistema de código aberto para o reconhecimento e tradução, em tempo real, do alfabeto manual da Língua Brasileira de Sinais (Libras). Utilizando técnicas de visão computacional e Redes Neurais Convolucionais (CNNs), a aplicação interpreta os gestos capturados por uma webcam e os converte para texto. O principal propósito é explorar o potencial da tecnologia como uma ferramenta para promover a acessibilidade e a inclusão social da comunidade surda.

## Metodologia e Funcionamento

O sistema opera a partir de um fluxo de etapas bem definidas. Primeiramente, realiza-se a **coleta de dados** com a gravação dos gestos. Em seguida, a biblioteca **MediaPipe** é utilizada para a **extração de parâmetros**, identificando os pontos de referência das mãos. Os dados extraídos passam por uma fase de **normalização e pré-processamento** para garantir a consistência.

Posteriormente, uma **Rede Neural Convolucional (CNN)** é submetida a **treinamento** com esses dados para aprender a classificar cada gesto. Por fim, o modelo treinado é integrado a uma aplicação que realiza a tradução em tempo real e exibe o resultado em uma interface gráfica. O processo culmina com **testes e avaliação** para aferir a robustez e a precisão do sistema.

## Tecnologias Utilizadas

  * **Linguagem de Programação:** Python
  * **Visão Computacional:** OpenCV, MediaPipe
  * **Aprendizado de Máquina:** TensorFlow (Keras)

## Como Executar

### Pré-requisitos

  * Python 3.9+
  * Git

### Instalação e Execução

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/...
    cd ...
    ```

2.  **Crie e ative um ambiente virtual:**

    ```bash
    # Crie o ambiente
    python -m venv .venv
    # Ative o ambiente (Windows)
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Instale as dependências necessárias:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação de tradução:**

    ```bash
    python src/predict.py
    ```

    *Pressione a tecla `ESC` para encerrar a aplicação.*

## Licença

Este projeto é disponibilizado como código aberto, incentivando o uso, a modificação e a distribuição pela comunidade para o avanço de tecnologias assistivas.
