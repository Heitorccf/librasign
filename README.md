# LibraSign - Sistema de Reconhecimento de Libras

## 📋 Sobre o Projeto

O LibraSign é um sistema de reconhecimento de gestos da Língua Brasileira de Sinais (Libras) desenvolvido com tecnologias de visão computacional e aprendizado profundo. O projeto utiliza redes neurais convolucionais para identificar e classificar gestos do alfabeto em Libras capturados através de uma webcam em tempo real.

Este sistema foi desenvolvido com o objetivo de contribuir para a acessibilidade e inclusão digital, oferecendo uma ferramenta que pode auxiliar no aprendizado e na comunicação através da Libras. A aplicação é capaz de reconhecer os gestos das letras do alfabeto, processando as imagens capturadas e fornecendo feedback visual instantâneo ao usuário.

## ✨ Funcionalidades Principais

O LibraSign oferece um conjunto completo de funcionalidades para captura, treinamento e reconhecimento de gestos:

**Captura de Dados**: O sistema permite a coleta sistemática de imagens de gestos através da webcam, organizando automaticamente o conjunto de dados por categoria alfabética. Durante a captura, o usuário pode visualizar em tempo real a detecção da mão e o processo de salvamento das imagens.

**Pré-processamento Inteligente**: Todas as imagens capturadas passam por um pipeline de processamento que inclui conversão para escala de cinza, redimensionamento padronizado e normalização dos valores de pixel, garantindo consistência e otimização para o treinamento do modelo.

**Treinamento de Modelo**: O sistema implementa uma arquitetura de rede neural convolucional otimizada para reconhecimento de padrões visuais, com camadas de convolução, pooling e regularização através de dropout para prevenir sobreajuste.

**Reconhecimento em Tempo Real**: A aplicação principal oferece predição instantânea dos gestos capturados pela webcam, exibindo o resultado diretamente na interface visual com indicadores claros do estado do sistema.

## 🛠️ Tecnologias Utilizadas

O projeto foi construído utilizando um conjunto robusto de bibliotecas e frameworks modernos:

- **Python 3.8+**: Linguagem principal do projeto
- **OpenCV**: Processamento de imagens e interface com webcam
- **MediaPipe**: Detecção e rastreamento de mãos em tempo real
- **TensorFlow/Keras**: Construção e treinamento da rede neural convolucional
- **NumPy**: Manipulação eficiente de arrays multidimensionais
- **Scikit-learn**: Ferramentas de pré-processamento e divisão de dados

## 📁 Estrutura do Projeto

```
librasign/
│
├── capture.py           # Sistema de captura de imagens via webcam
├── normalizing.py       # Pipeline de pré-processamento de dados
├── train.py            # Treinamento do modelo de rede neural
├── predict.py          # Aplicação de reconhecimento em tempo real
│
├── data/
│   └── raw/            # Diretório para armazenamento das imagens capturadas
│       ├── A/          # Imagens da letra A
│       ├── B/          # Imagens da letra B
│       └── ...         # Demais letras do alfabeto
│
├── models/
│   └── best_model.keras  # Modelo treinado salvo
│
├── LICENSE             # Licença GPL v3
└── README.md          # Este arquivo
```

## 📋 Pré-requisitos

Antes de iniciar a instalação do LibraSign, certifique-se de que seu sistema atende aos seguintes requisitos:

- Python 3.8 ou superior instalado
- Webcam funcional conectada ao computador
- Sistema operacional: Windows, Linux ou macOS
- Pelo menos 4GB de RAM disponível
- Espaço em disco: aproximadamente 500MB para o projeto e dados

## 🚀 Instalação

Siga este passo a passo detalhado para configurar o LibraSign em seu ambiente:

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/librasign.git
cd librasign
```

### 2. Crie um ambiente virtual

É altamente recomendado utilizar um ambiente virtual para evitar conflitos entre dependências:

```bash
# No Windows
python -m venv venv
venv\Scripts\activate

# No Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn
```

### 4. Crie a estrutura de diretórios necessária

```bash
mkdir -p data/raw models
```

## 💻 Como Usar

O LibraSign funciona através de um fluxo de trabalho sequencial que você deve seguir para obter os melhores resultados:

### Etapa 1: Captura de Dados

Execute o script de captura para coletar imagens de treinamento:

```bash
python capture.py
```

Durante a execução, o sistema mostrará uma janela com o vídeo da webcam. Para capturar imagens de uma letra específica, pressione a tecla correspondente (A-Z) no teclado. O sistema começará a salvar automaticamente as imagens detectadas da sua mão fazendo o gesto. Recomenda-se capturar pelo menos 100 imagens por letra, variando a posição, iluminação e ângulo da mão para criar um conjunto de dados robusto.

### Etapa 2: Processamento dos Dados

O pré-processamento é executado automaticamente quando você treina o modelo, mas você pode verificar se os dados estão corretos executando:

```bash
python normalizing.py
```

Este script carregará todas as imagens capturadas, aplicará as transformações necessárias e exibirá informações sobre o conjunto de dados, incluindo o número total de imagens e as classes detectadas.

### Etapa 3: Treinamento do Modelo

Inicie o processo de treinamento da rede neural:

```bash
python train.py
```

O treinamento pode levar alguns minutos, dependendo da quantidade de dados e do poder de processamento do seu computador. Durante o processo, você verá informações sobre o progresso, incluindo a acurácia do modelo em cada época. O melhor modelo será salvo automaticamente no diretório `models/`.

### Etapa 4: Reconhecimento em Tempo Real

Após o treinamento bem-sucedido, execute a aplicação principal:

```bash
python predict.py
```

Uma janela será aberta mostrando o vídeo da webcam com overlay de detecção. Faça gestos de letras em Libras em frente à câmera e o sistema exibirá a letra reconhecida em tempo real no canto superior direito da tela.

## 🧠 Como Funciona

O LibraSign implementa um pipeline completo de visão computacional e aprendizado de máquina que pode ser compreendido em quatro componentes principais:

**Detecção de Mãos**: O sistema utiliza o MediaPipe, uma biblioteca desenvolvida pelo Google, para detectar e rastrear pontos de referência anatômicos da mão em tempo real. O MediaPipe identifica 21 pontos-chave na mão, permitindo o cálculo preciso da região de interesse (ROI) que contém o gesto.

**Pré-processamento de Imagens**: Cada imagem capturada passa por uma série de transformações essenciais. Primeiro, a região da mão é extraída e convertida para escala de cinza, removendo informações de cor que são irrelevantes para a forma do gesto. Em seguida, a imagem é redimensionada para 224x224 pixels e os valores dos pixels são normalizados para o intervalo [0, 1], otimizando o processo de aprendizagem da rede neural.

**Arquitetura da Rede Neural**: O modelo utiliza uma arquitetura convolucional com duas camadas de convolução (32 e 64 filtros), intercaladas com camadas de max pooling para redução dimensional. Após o achatamento dos mapas de características, uma camada densa com 128 neurônios processa as informações, seguida de dropout (50%) para regularização. A camada de saída utiliza ativação softmax para gerar probabilidades para cada classe.

**Inferência e Predição**: Durante o reconhecimento em tempo real, cada frame capturado pela webcam passa pelo mesmo pipeline de pré-processamento usado no treinamento. O modelo processa a imagem e retorna um vetor de probabilidades, onde cada posição corresponde a uma letra do alfabeto. A letra com maior probabilidade é selecionada como a predição final.

## 🤝 Contribuindo

Contribuições são extremamente bem-vindas e valorizadas! O LibraSign é um projeto de código aberto e sua evolução depende da colaboração da comunidade. Se você deseja contribuir, siga estas diretrizes:

1. Faça um fork do projeto através do GitHub
2. Crie uma branch para sua funcionalidade (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças com mensagens descritivas (`git commit -m 'Adicionando nova funcionalidade X'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request detalhando as mudanças propostas

Algumas áreas onde contribuições seriam especialmente valiosas:
- Implementação de reconhecimento de palavras completas
- Otimização do modelo para melhor acurácia
- Interface gráfica mais elaborada
- Suporte para reconhecimento de números e expressões
- Documentação adicional e tutoriais

## 📄 Licença

Este projeto está licenciado sob a GNU General Public License v3.0 (GPL-3.0). Isso significa que você tem a liberdade de usar, modificar e distribuir este software, desde que mantenha a mesma licença e os créditos originais. Para mais detalhes, consulte o arquivo [LICENSE](LICENSE) no repositório.

A escolha da GPL v3 reflete nosso compromisso com o software livre e a acessibilidade tecnológica, garantindo que melhorias e derivações deste projeto permaneçam abertas e acessíveis à comunidade.

## ✉️ Autor e Contato

**Heitor Câmara Costa Fernandes**

- Email: Heitorccfernandes550@gmail.com
- Ano de desenvolvimento: 2025

Para dúvidas, sugestões ou reportar problemas, sinta-se à vontade para abrir uma issue no repositório ou entrar em contato diretamente através do email fornecido.

---

*Este projeto foi desenvolvido com o objetivo de promover a inclusão e acessibilidade através da tecnologia. Que ele possa contribuir para quebrar barreiras de comunicação e aproximar pessoas.*