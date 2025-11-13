Aqui está o texto formatado em Markdown para o seu `README.md`.

-----

# LibraSign

## Índice

## Índice

* [Introdução](#introducao)
* [Fundamentação Acadêmica](#fundamentacao-academica)
* [Visão Geral do Sistema](#visao-geral-do-sistema)
    * [Escopo e Limitações](#escopo-e-limitacoes)
    * [Arquitetura do Sistema](#arquitetura-do-sistema)
    * [Fluxo de Processamento](#fluxo-de-processamento)
* [Requisitos do Sistema](#requisitos-do-sistema)
    * [Requisitos de Software](#requisitos-de-software)
    * [Requisitos de Hardware](#requisitos-de-hardware)
* [Guia de Instalação](#guia-de-instalacao)
    * [Etapa 1: Preparação do Ambiente](#etapa-1-preparacao-do-ambiente)
    * [Etapa 2: Clonagem do Repositório](#etapa-2-clonagem-do-repositorio)
    * [Etapa 3: Configuração do Ambiente Virtual](#etapa-3-configuracao-do-ambiente-virtual)
    * [Etapa 4: Instalação das Dependências](#etapa-4-instalacao-das-dependencias)
* [Execução do Sistema](#execucao-do-sistema)
    * [Modo de Uso Padrão](#modo-de-uso-padrao)
    * [Captura de Novo Dataset (Opcional)](#captura-de-novo-dataset-opcional)
    * [Retreinamento do Modelo](#retreinamento-do-modelo)
* [Dataset Público](#dataset-publico)
* [Solução de Problemas](#solucao-de-problemas)
* [Aplicabilidade e Extensibilidade](#aplicabilidade-e-extensibilidade)
* [Considerações Finais](#consideracoes-finais)
* [Referências e Documentação Complementar](#referencias-e-documentacao-complementar)

-----

## Introdução

O **LibraSign** é um sistema de código aberto desenvolvido como projeto de conclusão de curso (TCC) que utiliza técnicas de visão computacional e aprendizado de máquina para realizar o reconhecimento em tempo real de gestos correspondentes ao alfabeto manual da Língua Brasileira de Sinais (**Libras**). O projeto foi concebido com o objetivo de explorar metodologias de processamento de dados geométricos e classificação neural para aplicações de acessibilidade comunicacional.

É fundamental compreender que o LibraSign possui um escopo deliberadamente restrito: o sistema reconhece *exclusivamente* as configurações de mão correspondentes às letras do alfabeto manual (A a Z), não sendo capaz de interpretar palavras completas, sinais compostos ou a gramática espacial complexa da Libras. Esta delimitação foi estabelecida para viabilizar uma investigação acadêmica aprofundada sobre a eficácia de redes neurais artificiais na classificação de gestos estáticos, servindo como prova de conceito para futuras expansões.

O projeto destina-se primariamente ao ambiente acadêmico e educacional, constituindo uma ferramenta de estudo sobre processamento de sinais visuais e aprendizado supervisionado. Embora funcional, o sistema não foi projetado para substituir intérpretes profissionais ou para uso comunicacional cotidiano em larga escala, visto que a Libras envolve elementos linguísticos complexos que ultrapassam o escopo deste trabalho, incluindo expressões faciais, movimentos corporais e estruturas gramaticais próprias.

-----

## Fundamentação Acadêmica

Este projeto representa a materialização de uma investigação científica rigorosa conduzida no âmbito do curso de Bacharelado em Sistemas de Informação. A fundamentação teórica completa, incluindo revisão de literatura sobre línguas de sinais, técnicas de visão computacional, arquiteturas de redes neurais, metodologia experimental, análise estatística dos resultados e discussão sobre as implicações sociais da tecnologia assistiva, encontra-se detalhadamente documentada no trabalho de conclusão de curso.

Para uma compreensão aprofundada dos fundamentos teóricos, das decisões arquiteturais, dos experimentos conduzidos e das conclusões alcançadas, recomenda-se enfaticamente a leitura do documento acadêmico completo, disponível neste repositório:

**📄 HeitorFernandes-TCC\_BSI.pdf**

O documento acadêmico aborda tópicos essenciais como a diferenciação entre a comunicação em línguas de sinais e a datilologia (soletração manual), as limitações das abordagens baseadas em processamento de imagens brutas, a escolha por representações geométricas de landmarks, e as métricas de desempenho obtidas através de validação cruzada estratificada.

-----

## Visão Geral do Sistema

### Escopo e Limitações

Antes de prosseguir com a utilização do sistema, é imperativo que o usuário compreenda claramente o escopo de funcionalidade do LibraSign. O sistema foi desenvolvido especificamente para reconhecer as configurações de mão estáticas do alfabeto manual da Libras, que correspondem às vinte e seis letras do alfabeto latino (A–Z). Esta escolha metodológica foi deliberada e alinha-se com os objetivos de pesquisa do projeto.

**O que o sistema reconhece:**

  * Configurações de mão correspondentes a cada uma das letras de A a Z do alfabeto manual de Libras, quando apresentadas de forma estática e isolada diante da câmera.

**O que o sistema NÃO reconhece:**

  * Palavras completas em Libras, que frequently são representadas por sinais únicos e não pela soletração letra a letra.
  * Sinais compostos ou ideográficos que constituem o vocabulário padrão da língua.
  * Expressões faciais, movimento corporal ou utilização do espaço de sinalização, elementos essenciais da gramática de Libras.
  * Variações regionais ou dialetos da língua de sinais.
  * Transições dinâmicas entre letras ou gestos em movimento contínuo.

Esta delimitação posiciona o LibraSign como uma ferramenta educacional e de pesquisa, adequada para o estudo de técnicas de reconhecimento de padrões e para aplicações didáticas de ensino do alfabeto manual, mas não como um tradutor completo da língua de sinais. O projeto estabelece fundamentos que podem ser expandidos em trabalhos futuros para incluir vocabulário mais amplo e elementos linguísticos adicionais.

### Arquitetura do Sistema

O LibraSign foi arquitetado seguindo uma metodologia modular que separa claramente as responsabilidades de cada componente do sistema. Esta organização facilita a manutenção, o teste e a eventual expansão das funcionalidades. A arquitetura compreende três módulos principais:

  * **Módulo de Captura de Dados (`src/capture.py`):** Este componente é responsável pela aquisição de dados de treinamento. Utilizando a biblioteca **MediaPipe** desenvolvida pelo Google, o módulo acessa a câmera do dispositivo e realiza a detecção em tempo real das mãos presentes no campo de visão. Para cada frame capturado, o MediaPipe identifica vinte e um pontos de referência anatômicos (landmarks) na mão detectada, extraindo suas coordenadas tridimensionais (x, y, z) no espaço normalizado. Estes dados geométricos, ao invés de imagens brutas em pixels, são persistidos em arquivos CSV organizados por classe, criando um dataset leve e estruturado que facilita o processamento posterior.

  * **Módulo de Treinamento (`src/train.py`):** Este componente implementa o pipeline completo de aprendizado supervisionado. Inicialmente, o módulo carrega o dataset de landmarks a partir dos arquivos CSV gerados na etapa de captura. Em seguida, aplica uma transformação de normalização geométrica que torna os dados invariantes à posição absoluta da mão no quadro e à escala (distância da câmera), centralizando os pontos em relação ao pulso e normalizando pelo comprimento característico da mão. Após a normalização, os dados são padronizados utilizando o `StandardScaler` para apresentarem média zero e variância unitária. O modelo escolhido é um **Perceptron Multicamadas (MLP)** com duas camadas ocultas, treinado através do algoritmo de retropropagação de gradientes. A avaliação do desempenho é conduzida através de validação cruzada estratificada com cinco partições, garantindo estimativas robustas da capacidade de generalização. Ao final, o modelo treinado, o objeto de padronização e o mapeamento de classes são serializados para uso na inferência.

  * **Módulo de Predição em Tempo Real (`src/predict.py`):** Este é o módulo de interface com o usuário, responsável pela aplicação prática do modelo treinado. O componente carrega os artefatos persistidos (modelo, scaler e classes), inicializa a captura de vídeo e processa cada frame em tempo real. Para cada detecção de mão, as coordenadas dos landmarks são extraídas, normalizadas e padronizadas exatamente da mesma forma que durante o treinamento, garantindo a consistência dos dados de entrada. O vetor resultante é submetido ao classificador neural, que retorna probabilidades para cada classe. O sistema implementa duas estratégias de estabilização: um filtro de votação majoritária sobre os últimos dez frames para suavizar predições ruidosas, e um mecanismo de confirmação temporal que exige que uma letra permaneça estável por dois segundos antes de ser adicionada à frase em construção. O resultado é apresentado visualmente na tela, juntamente com indicadores de confiança e a sentença formada.

### Fluxo de Processamento

Para auxiliar na compreensão da operação do sistema, apresenta-se a seguir o fluxo sequencial de processamento desde a captura do gesto até a apresentação do resultado:

1.  **Aquisição do Frame:** O sistema captura continuamente frames da câmera do dispositivo em tempo real, processando aproximadamente vinte quadros por segundo dependendo do hardware disponível.
2.  **Detecção da Mão:** Cada frame é processado pelo modelo de detecção de mãos do MediaPipe, que utiliza redes neurais convolucionais leves para identificar a presença e localização de mãos na imagem. Quando uma mão é detectada com confiança superior ao limiar estabelecido, o sistema prossegue para a extração de landmarks.
3.  **Extração de Landmarks:** O MediaPipe identifica vinte e um pontos anatômicos na mão detectada, correspondendo a localizações como a ponta de cada dedo, as articulações metacarpofalângicas, interfalângicas proximais e distais, além do pulso. Cada landmark é representado por suas coordenadas tridimensionais normalizadas em relação às dimensões da imagem.
4.  **Normalização Geométrica:** Os landmarks brutos são transformados para garantir invariância. Primeiro, todos os pontos são transladados para que o pulso (landmark zero) fique na origem do sistema de coordenadas. Em seguida, calcula-se a distância euclidiana entre o pulso e a base do dedo médio (landmark nove), utilizando esta medida como fator de escala. Todos os pontos são então divididos por este fator, resultando em uma representação onde o tamanho e a posição absoluta da mão não influenciam a classificação.
5.  **Padronização Estatística:** Os dados normalizados geometricamente são padronizados utilizando o `StandardScaler` treinado, que subtrai a média e divide pelo desvio padrão de cada característica, conforme calculado no conjunto de treinamento. Esta etapa garante que todas as dimensões do vetor de entrada contribuam de forma equilibrada para a decisão do classificador.
6.  **Classificação Neural:** O vetor de características padronizado é propagado através das camadas do MLP. A rede processa a informação através de suas cento e vinte e oito unidades na primeira camada oculta, seguidas por sessenta e quatro unidades na segunda camada, aplicando funções de ativação não-lineares. A camada de saída, com dimensionalidade igual ao número de classes, produz probabilidades através de uma função softmax.
7.  **Estabilização Temporal:** Para reduzir oscilações e predições espúrias, o sistema mantém um histórico das últimas dez predições e aplica votação majoritária. Além disso, uma letra só é considerada confirmada se permanecer como predição predominante por dois segundos consecutivos, evitando que movimentos transitórios sejam interpretados como gestos intencionais.
8.  **Apresentação dos Resultados:** O sistema renderiza sobre o frame de vídeo os landmarks detectados, a letra atualmente reconhecida com indicação de confiança, uma barra de progresso para confirmação temporal e, na porção inferior da tela, a sentença formada pelas letras confirmadas até o momento.

-----

## Requisitos do Sistema

### Requisitos de Software

Para a correta execução do LibraSign, é necessário que o ambiente de desenvolvimento atenda aos seguintes requisitos de software:

  * **Sistema Operacional:** O sistema foi desenvolvido e testado em ambientes Linux (distribuições baseadas em Debian e Fedora), macOS (versões 11 e superiores) e Windows 10/11. Em teoria, qualquer sistema operacional que suporte Python e as bibliotecas necessárias deve ser capaz de executar o software.
  * **Interpretador Python:** É imprescindível a instalação do Python na versão **3.11.13**, conforme especificado no desenvolvimento do projeto. Versões anteriores à 3.9 não são suportadas devido à utilização de recursos sintáticos e de biblioteca introduzidos nessas versões mais recentes. Versões posteriores à 3.11.13 podem funcionar, mas não foram extensivamente testadas e podem apresentar incompatibilidades com algumas dependências.
  * **Gerenciador de Pacotes pip:** A instalação das dependências do projeto é realizada através do `pip`, o gerenciador de pacotes padrão do Python. Versões recentes do Python já incluem o `pip`, mas é recomendável verificar sua presença e atualização antes de prosseguir.
  * **Câmera Funcional:** O sistema requer acesso a uma câmera (webcam integrada ou externa) para captura de vídeo em tempo real. Certifique-se de que o sistema operacional concedeu as permissões necessárias para que aplicações acessem a câmera.
  * **Bibliotecas Python Essenciais:** As seguintes bibliotecas constituem o núcleo funcional do sistema e suas versões específicas são críticas para o funcionamento adequado:

<!-- end list -->

```
# Versão do Python recomendada para este projeto: 3.11.13

scikit-learn==1.7.2
numpy==2.2.6
pandas==2.3.2
opencv-python==4.12.0.88
mediapipe==0.10.14
kagglehub==0.3.13
```

### Requisitos de Hardware

Embora o LibraSign tenha sido otimizado para execução em hardware modesto, certos requisitos mínimos devem ser atendidos para garantir desempenho adequado:

  * **Processador:** Recomenda-se um processador moderno com pelo menos dois núcleos físicos operando a uma frequência base de 2.0 GHz ou superior. Processadores Intel Core i3 de oitava geração ou superiores, AMD Ryzen 3 ou equivalentes são adequados. O sistema foi testado com sucesso em processadores Intel Core i5 e i7, bem como em processadores ARM de dispositivos Apple Silicon.
  * **Memória RAM:** Um mínimo de 4 GB de RAM é necessário para a execução do sistema. No entanto, recomenda-se 8 GB ou mais para operação confortável, especialmente durante o treinamento do modelo, que pode consumir memória significativa dependendo do tamanho do dataset.
  * **Armazenamento:** O projeto em si ocupa aproximadamente 16.6 MB de espaço em disco. O dataset público de landmarks, quando baixado, adiciona cerca de 32.39 MB. Recomenda-se ter pelo menos 1 GB de espaço livre em disco para acomodar o projeto, datasets, modelos treinados e quaisquer datasets adicionais que o usuário deseje capturar.
  * **Câmera:** É essencial uma câmera com resolução mínima de 640x480 pixels (VGA) e taxa de captura de pelo menos 15 frames por segundo. Câmeras com resolução HD (1280x720) ou superior e taxas de 30 FPS proporcionam melhor experiência de uso. A câmera deve estar posicionada de forma a capturar claramente a mão do usuário contra um fundo relativamente uniforme, preferencialmente com boa iluminação ambiente.
  * **Iluminação:** Embora não seja um requisito de hardware per se, condições adequadas de iluminação são cruciais para o desempenho do sistema. Recomenda-se ambiente bem iluminado, preferencialmente com luz natural difusa ou iluminação artificial uniforme, evitando contraluz intenso ou sombras fortes que possam dificultar a detecção dos landmarks pela biblioteca MediaPipe.
  * **Sistema Gráfico:** Embora o sistema não exija GPU dedicada, é necessário suporte básico para exibição de janelas gráficas e renderização de vídeo. Em sistemas Linux, certifique-se de que o servidor X (X11) ou Wayland esteja configurado corretamente. Em ambientes de servidor ou contêineres sem interface gráfica, o sistema não funcionará adequadamente.

-----

## Guia de Instalação

Este guia conduzirá o usuário através de todas as etapas necessárias para preparar o ambiente de desenvolvimento e instalar o LibraSign em seu sistema. As instruções são apresentadas de forma detalhada e incluem comandos específicos para diferentes sistemas operacionais quando aplicável.

### Etapa 1: Preparação do Ambiente

Antes de iniciar a instalação do LibraSign propriamente dito, é necessário garantir que o interpretador Python esteja instalado e configurado corretamente no sistema.

**Verificação da Instalação do Python:**

Primeiramente, verifique se o Python está instalado no sistema e qual versão está disponível. Abra o terminal (Linux/macOS) ou Prompt de Comando/PowerShell (Windows) e execute o seguinte comando:

```bash
python --version
```

Em alguns sistemas, especialmente Linux e macOS, pode ser necessário utilizar explicitamente o comando `python3`:

```bash
python3 --version
```

O comando deve retornar a versão do Python instalada, idealmente `Python 3.11.13` ou uma versão 3.11.x. Se a versão retornada for inferior a 3.9, será necessário atualizar o Python. Se o comando não for reconhecido, o Python não está instalado ou não está configurado corretamente no PATH do sistema.

**Instalação do Python (se necessário):**

Caso o Python não esteja instalado, siga as instruções específicas para seu sistema operacional:

  * **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update
    sudo apt install python3.11 python3.11-venv python3-pip
    ```
  * **Linux (Fedora):**
    ```bash
    sudo dnf install python3.11
    ```
  * **macOS:** Recomenda-se utilizar o Homebrew para instalação:
    ```bash
    brew install python@3.11
    ```
  * **Windows:** Baixe o instalador oficial do Python 3.11 através do site `python.org`, certificando-se de marcar a opção "**Add Python to PATH**" durante a instalação.

**Verificação do pip:**

O `pip` deve ser instalado automaticamente junto com o Python. Verifique sua presença e versão:

```bash
python -m pip --version
```

ou

```bash
python3 -m pip --version
```

Se o `pip` não estiver disponível, ele pode ser instalado através do script `get-pip.py` disponível no site oficial do Python.

### Etapa 2: Clonagem do Repositório

Com o Python devidamente instalado, o próximo passo consiste em obter uma cópia local do repositório do LibraSign. Certifique-se de ter o **Git** instalado no sistema antes de prosseguir.

**Verificação do Git:**

```bash
git --version
```

Se o Git não estiver instalado, visite `git-scm.com` e siga as instruções para seu sistema operacional.

**Clonagem do Repositório:**

Navegue até o diretório onde deseja armazenar o projeto e execute o seguinte comando para clonar o repositório:

```bash
git clone https://github.com/heitorccf/librasign.git
```

Aguarde enquanto o Git baixa todos os arquivos do repositório. Ao concluir, um novo diretório chamado `librasign` será criado contendo todos os arquivos do projeto.

**Navegação até o Diretório do Projeto:**

Entre no diretório recém-criado:

```bash
cd librasign
```

Todos os comandos subsequentes devem ser executados a partir deste diretório raiz do projeto.

### Etapa 3: Configuração do Ambiente Virtual

A utilização de um ambiente virtual Python é uma prática altamente recomendada e considerada essencial para o desenvolvimento de projetos Python. O ambiente virtual cria um contexto isolado onde as dependências do projeto podem ser instaladas sem interferir com outros projetos ou com as bibliotecas do sistema. Esta abordagem previne conflitos de versão e facilita a reprodutibilidade do ambiente de execução.

**Criação do Ambiente Virtual:**

  * **Linux e macOS:**
    ```bash
    python3 -m venv .venv
    ```
  * **Windows:**
    ```bash
    python -m venv .venv
    ```

Este comando cria um diretório chamado `.venv` dentro do diretório do projeto, contendo uma cópia isolada do interpretador Python e ferramentas associadas. O ponto inicial no nome (`.venv`) é uma convenção que indica um diretório oculto ou de configuração.

**Ativação do Ambiente Virtual:**

Após criar o ambiente virtual, é necessário ativá-lo. O comando de ativação varia conforme o sistema operacional:

  * **Linux e macOS:**
    ```bash
    source .venv/bin/activate
    ```
  * **Windows (Prompt de Comando):**
    ```bash
    .venv\Scripts\activate.bat
    ```
  * **Windows (PowerShell):**
    ```bash
    .venv\Scripts\Activate.ps1
    ```

*Nota importante para usuários do Windows PowerShell:* Caso encontre um erro relacionado à política de execução de scripts, você pode precisar alterar temporariamente a política executando o PowerShell como administrador e digitando:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Após a ativação bem-sucedida, você notará que o prompt do terminal foi modificado, exibindo o nome do ambiente virtual entre parênteses, geralmente `(.venv)` antes do caminho do diretório. Esta modificação visual confirma que o ambiente virtual está ativo e que quaisquer pacotes instalados via `pip` serão instalados neste ambiente isolado.

**Desativação do Ambiente Virtual (para referência futura):**

Embora não seja necessário desativar o ambiente imediatamente, é útil saber que, quando desejar sair do ambiente virtual, basta executar:

```bash
deactivate
```

Este comando funciona em todos os sistemas operacionais e retorna o terminal ao ambiente Python global do sistema.

### Etapa 4: Instalação das Dependências

Com o ambiente virtual ativado, procede-se à instalação de todas as bibliotecas necessárias para a execução do LibraSign. O projeto inclui um arquivo `requirements.txt` que lista todas as dependências com suas versões específicas, facilitando a instalação em uma única operação.

**Atualização do pip (recomendado):**

Antes de instalar as dependências, é prudente garantir que o `pip` está atualizado para sua versão mais recente:

```bash
python -m pip install --upgrade pip
```

**Instalação das Dependências:**

Execute o seguinte comando para instalar todas as bibliotecas listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

O `pip` processará o arquivo, resolverá as dependências, baixará os pacotes necessários dos repositórios oficiais do Python Package Index (PyPI) e os instalará no ambiente virtual. Este processo pode levar alguns minutos dependendo da velocidade da conexão com a internet e do poder de processamento do sistema.

Durante a instalação, você verá mensagens indicando o progresso do download e instalação de cada pacote. É normal que alguns pacotes apresentem mensagens de compilação, especialmente em sistemas Linux, caso seja necessário compilar extensões em C. Aguarde até que o processo seja concluído completamente.

**Verificação da Instalação:**

Após a conclusão da instalação, é recomendável verificar se as principais bibliotecas foram instaladas corretamente. Você pode listar todas as bibliotecas instaladas com:

```bash
pip list
```

Procure na lista as bibliotecas essenciais: `scikit-learn`, `numpy`, `pandas`, `opencv-python`, `mediapipe` e `kagglehub`. Se todas estiverem presentes, a instalação foi bem-sucedida.

Alternativamente, você pode testar a importação das bibliotecas diretamente no interpretador Python:

```bash
python -c "import cv2, mediapipe, sklearn, numpy, pandas; print('Todas as bibliotecas foram importadas com sucesso')"
```

Se o comando for executado sem erros e exibir a mensagem de sucesso, o ambiente está corretamente configurado e pronto para uso.

-----

## Execução do Sistema

Uma vez que o ambiente está devidamente preparado e todas as dependências foram instaladas, o usuário pode proceder à execução do LibraSign. É importante compreender que o sistema oferece diferentes modos de operação, cada um atendendo a propósitos específicos.

### Modo de Uso Padrão

Para a maioria dos usuários interessados em experimentar o sistema de reconhecimento de gestos sem a necessidade de treinar um novo modelo, o modo de uso padrão é o mais apropriado. Este modo utiliza um modelo pré-treinado que foi desenvolvido com o dataset público disponível no Kaggle.

**Execução do Tradutor em Tempo Real:**

Com o ambiente virtual ativado, execute o script de predição:

```bash
python src/predict.py
```

Ao executar este comando, o sistema realizará as seguintes operações:

  * Primeiro, carregará os artefatos do modelo treinado a partir do diretório `models/`, incluindo o classificador MLP serializado, o objeto `StandardScaler` utilizado para padronização dos dados e o mapeamento de classes que relaciona os índices numéricos às letras do alfabeto.
  * Em seguida, inicializará a biblioteca MediaPipe para detecção de mãos e configurará os parâmetros de confiança para detecção e rastreamento.
  * Finalmente, abrirá uma janela gráfica exibindo o feed de vídeo da câmera do dispositivo, com sobreposição dos landmarks detectados, a letra atualmente reconhecida e a frase em construção.

**Instruções de Uso Durante a Execução:**

1.  Posicione sua mão dominante no campo de visão da câmera, a uma distância aproximada de trinta a sessenta centímetros, contra um fundo relativamente uniforme. A iluminação adequada é fundamental para a detecção correta dos landmarks.
2.  Forme com a mão a configuração correspondente a uma letra do alfabeto manual de Libras. Mantenha a posição estável e aguarde enquanto o sistema processa os frames. Você observará uma barra de progresso verde na tela que indica o tempo restante para confirmação da letra.
3.  Após dois segundos com a mesma letra sendo detectada consistentemente, ela será adicionada à frase exibida na parte inferior da tela. Continue formando as letras subsequentes para construir palavras ou frases.

**Controles do Teclado:**

Durante a execução do sistema, as seguintes teclas de controle estão disponíveis:

  * **ESC (Escape):** Encerra a aplicação e fecha a janela de vídeo.
  * **Backspace:** Remove a última letra adicionada à frase, permitindo correção de erros.
  * **C (letra cê):** Limpa completamente a frase em construção, permitindo recomeçar.

**Encerramento do Sistema:**

Para encerrar o sistema adequadamente, pressione a tecla **ESC**. O script liberará os recursos da câmera e fechará todas as janelas gráficas abertas, retornando ao prompt do terminal.

### Captura de Novo Dataset (Opcional)

Usuários interessados em experimentar com diferentes conjuntos de dados, em expandir o sistema para reconhecer gestos adicionais ou em coletar dados específicos de suas próprias configurações de mão podem utilizar o script de captura para gerar um dataset personalizado.

**Execução do Script de Captura:**

Com o ambiente virtual ativado, execute:

```bash
python src/capture.py
```

O sistema abrirá uma janela de vídeo e aguardará instruções do usuário através do teclado.

**Processo de Captura de Dados:**

O script permite que você capture amostras organizadas por classe (letra). Para iniciar a captura de uma letra específica:

1.  Pressione a tecla correspondente à letra que deseja capturar (A a Z). O sistema iniciará imediatamente a captura automática de landmarks sempre que uma mão for detectada.
2.  Forme a configuração de mão correspondente à letra escolhida e mantenha-a relativamente estável enquanto movimenta levemente a mão, alterando sutilmente sua posição, rotação e distância da câmera. Esta variação é importante para que o modelo aprenda a reconhecer a letra em diferentes condições.
3.  O sistema capturará automaticamente até mil amostras para cada letra, salvando os dados no diretório `data/landmarks/` em arquivos CSV nomeados conforme a letra (por exemplo, `A.csv`, `B.csv`).
4.  A tela exibirá o progresso da captura, indicando quantas amostras já foram coletadas do total de mil.

**Controles Durante a Captura:**

  * **A-Z:** Inicia ou alterna a captura para a letra pressionada.
  * **0 (zero):** Captura amostras da classe "nenhum", representando quadros onde nenhuma letra específica está sendo formada.
  * **Espaço:** Pausa ou retoma a captura para a classe atual.
  * **ESC:** Encerra o script de captura.

**Considerações Importantes:**

  * Para obter um modelo robusto, é fundamental capturar amostras com variabilidade adequada. Varie a iluminação, o ângulo da câmera, a rotação da mão e a distância durante a captura. Capture amostras de diferentes pessoas se possível, pois isso aumenta a capacidade de generalização do modelo.
  * Certifique-se de formar corretamente cada configuração de mão conforme o alfabeto manual de Libras. Configurações incorretas durante a captura resultarão em dados rotulados erroneamente, prejudicando significativamente o desempenho do modelo treinado.

### Retreinamento do Modelo

Após capturar um dataset personalizado ou modificado, é necessário treinar um novo modelo neural para incorporar esses dados.

**Execução do Script de Treinamento:**

Com o ambiente virtual ativado, execute:

```bash
python src/train.py
```

O script realizará as seguintes operações de forma automatizada:

1.  Primeiramente, baixará o dataset público de referência a partir do Kaggle utilizando a biblioteca `kagglehub`. Este dataset serve como base de treinamento padrão.
2.  Em seguida, carregará todos os arquivos CSV do diretório de landmarks, construindo as matrizes de características e vetores de rótulos.
3.  Aplicará a normalização geométrica aos landmarks, tornando os dados invariantes à posição e escala da mão.
4.  Executará o processo de validação cruzada estratificada com cinco partições, treinando e avaliando o modelo MLP em cada partição, reportando a acurácia obtida.
5.  Calculará e exibirá a acurácia média e o desvio padrão através das partições, fornecendo uma estimativa robusta do desempenho esperado.
6.  Gerará e salvará uma matriz de confusão que detalha os padrões de acerto e erro do classificador.
7.  Finalmente, treinará um modelo definitivo utilizando todos os dados disponíveis e salvará os artefatos (modelo, scaler e classes) no diretório `models/`.

**Interpretação dos Resultados:**

Durante o treinamento, o sistema exibirá mensagens indicando o progresso e os resultados de cada fold da validação cruzada. Preste atenção à acurácia reportada, que idealmente deve estar acima de noventa por cento para um desempenho satisfatório na aplicação prática.

A matriz de confusão salva pode ser analisada posteriormente para identificar quais letras o modelo confunde com maior frequência, informando possíveis melhorias no dataset ou na arquitetura do modelo.

**Duração do Treinamento:**

O tempo necessário para o treinamento completo varia conforme o tamanho do dataset e a capacidade de processamento do hardware. Em um computador com processador moderno, o treinamento típico com o dataset padrão (aproximadamente vinte e sete mil amostras) leva entre dois e cinco minutos. Datasets maiores ou hardware mais modesto podem requerer tempo adicional.

-----

## Dataset Público

Como parte do compromisso com a ciência aberta e a reprodutibilidade da pesquisa, o dataset de landmarks utilizado no desenvolvimento e treinamento do LibraSign foi disponibilizado publicamente na plataforma Kaggle. Este dataset contém aproximadamente mil amostras para cada uma das vinte e sete classes (vinte e seis letras mais a classe "nenhum"), totalizando cerca de vinte e sete mil exemplos de configurações de mão.

O dataset pode ser acessado, visualizado e baixado através do seguinte link:

🔗 **[Libras Landmark Dataset (A-Z) no Kaggle](https://www.kaggle.com/datasets/heitorccf/librasign)**

**Estrutura do Dataset:**

O dataset consiste em arquivos CSV, um para cada classe, onde cada linha representa uma amostra individual contendo sessenta e três valores numéricos de ponto flutuante. Estes valores correspondem às coordenadas `x`, `y` e `z` dos vinte e um landmarks extraídos pelo MediaPipe, organizados sequencialmente (`x₀, y₀, z₀, x₁, y₁, z₁, ..., x₂₀, y₂₀, z₂₀`).

**Utilização do Dataset:**

Pesquisadores e desenvolvedores interessados em trabalhos relacionados podem utilizar este dataset para:

  * Reproduzir os resultados apresentados no trabalho de conclusão de curso.
  * Explorar diferentes arquiteturas de redes neurais e técnicas de classificação.
  * Desenvolver sistemas de reconhecimento de gestos baseados em landmarks.
  * Realizar análises comparativas de desempenho entre diferentes abordagens metodológicas.
  * Expandir o sistema com classes adicionais ou datasets complementares.

Ao utilizar este dataset em trabalhos acadêmicos ou projetos, solicita-se a devida citação conforme as práticas acadêmicas estabelecidas.

-----

## Solução de Problemas

Durante a instalação ou execução do LibraSign, alguns problemas podem ser encontrados dependendo das especificidades do sistema operacional, da configuração do hardware ou de variações no ambiente de software. Esta seção documenta os problemas mais comuns e suas respectivas soluções.

**Problema: Comando `python` não reconhecido ou versão incorreta do Python**

  * *Sintomas:* Ao executar `python --version`, o terminal retorna um erro indicando que o comando não foi encontrado, ou retorna uma versão do Python 2.x.
  * *Causa:* Em muitos sistemas Unix-like (Linux e macOS), o comando `python` aponta para o Python 2.x por razões de compatibilidade histórica, enquanto o Python 3.x deve ser invocado explicitamente através do comando `python3`.
  * *Solução:* Em todos os comandos apresentados neste guia onde aparece `python`, substitua por `python3`. Por exemplo, ao invés de `python src/predict.py`, utilize `python3 src/predict.py`. Alternativamente, você pode criar um alias em seu shell ou modificar as variáveis de ambiente do sistema para que o comando `python` aponte para o Python 3.

**Problema: Erro "Permission denied" ao tentar acessar a câmera**

  * *Sintomas:* O sistema inicia mas não consegue abrir a câmera, exibindo a mensagem "[ERRO] Não foi possível abrir a câmera, verifique a conexão e as permissões".
  * *Causa:* O sistema operacional está bloqueando o acesso da aplicação à câmera por questões de privacidade e segurança.
  * *Solução no macOS:* Navegue até *Preferências do Sistema \> Segurança e Privacidade \> Privacidade \> Câmera*, e certifique-se de que o *Terminal* (ou o aplicativo através do qual você está executando o Python) tem permissão para acessar a câmera.
  * *Solução no Windows:* Vá para *Configurações \> Privacidade \> Câmera*, e verifique se "Permitir que aplicativos acessem sua câmera" está ativado. Certifique-se também de que "Aplicativos da área de trabalho" tem permissão.
  * *Solução no Linux:* Verifique se seu usuário pertence ao grupo `video`. Execute `groups` no terminal e verifique se `video` está listado. Se não estiver, adicione seu usuário ao grupo com `sudo usermod -a -G video $USER` e reinicie a sessão.

**Problema: `ImportError` ao tentar importar `cv2`, `mediapipe` ou outras bibliotecas**

  * *Sintomas:* Ao executar qualquer script, o Python retorna um erro similar a "ModuleNotFoundError: No module named 'cv2'" ou similar para outras bibliotecas.
  * *Causa:* As dependências não foram instaladas corretamente no ambiente virtual, ou o ambiente virtual não está ativado.
  * *Solução:* Primeiro, certifique-se de que o ambiente virtual está ativado verificando se há o prefixo `(.venv)` no prompt do terminal. Se não estiver ativado, execute o comando de ativação apropriado para seu sistema. Em seguida, execute novamente `pip install -r requirements.txt` para garantir que todas as dependências sejam instaladas. Se o problema persistir, tente desinstalar e reinstalar a biblioteca problemática especificamente, por exemplo: `pip uninstall opencv-python` seguido de `pip install opencv-python==4.12.0.88`.

**Problema: MediaPipe não detecta a mão ou detecção é instável**

  * *Sintomas:* O sistema executa, mas os landmarks da mão não são desenhados na tela, ou a detecção é intermitente e instável.
  * *Causa:* Condições inadequadas de iluminação, fundo muito complexo ou confuso, ou a mão está muito próxima ou muito distante da câmera.
  * *Solução:* Melhore a iluminação do ambiente, preferencialmente utilizando luz natural difusa ou iluminação artificial uniforme. Posicione-se contra um fundo simples e de cor relativamente uniforme, evitando padrões complexos ou elementos que possam ser confundidos com a mão. Ajuste a distância entre sua mão e a câmera, experimentando posições entre trinta e sessenta centímetros. Certifique-se de que sua mão está completamente visível no quadro, não sendo cortada pelas bordas da imagem.

**Problema: Modelo apresenta acurácia baixa ou predições inconsistentes**

  * *Sintomas:* Durante o uso do sistema em tempo real, as predições parecem aleatórias ou frequentemente incorretas, ou durante o treinamento a acurácia reportada é significativamente inferior a noventa por cento.
  * *Causa:* Dataset de treinamento com problemas, como dados rotulados incorretamente, insuficiente variabilidade nas amostras ou quantidade inadequada de exemplos por classe.
  * *Solução:* Revise o processo de captura do dataset, certificando-se de que as configurações de mão estão corretas conforme o alfabeto manual de Libras. Capture mais amostras para cada classe, garantindo variabilidade em termos de iluminação, ângulo, rotação e distância da câmera. Considere envolver múltiplas pessoas na captura de dados para aumentar a diversidade. Se estiver utilizando o dataset público, verifique se o download foi completo e se os arquivos não estão corrompidos.

**Problema: O script de treinamento falha ao baixar o dataset do Kaggle**

  * *Sintomas:* Durante a execução de `train.py`, o sistema reporta erros relacionados ao `kagglehub` ou falha ao baixar o dataset.
  * *Causa:* Problemas de conectividade com a internet, firewall bloqueando a conexão, ou configuração inadequada das credenciais do Kaggle.
  * *Solução:* Verifique sua conexão com a internet e tente novamente. Se estiver atrás de um firewall corporativo, pode ser necessário configurar proxies. Alternativamente, você pode baixar o dataset manualmente através do link do Kaggle, descompactar e colocar os arquivos CSV no diretório `data/landmarks/`, em seguida modificar o script `train.py` para não executar o download automático (comentando a seção de download do `kagglehub`).

**Problema: No Windows, erro "cannot be loaded because running scripts is disabled on this system" ao ativar o ambiente virtual**

  * *Sintomas:* Ao tentar ativar o ambiente virtual no PowerShell, aparece uma mensagem de erro relacionada à política de execução de scripts.
  * *Causa:* O PowerShell tem uma política de segurança que por padrão impede a execução de scripts não assinados.
  * *Solução:* Abra o PowerShell como administrador e execute `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. Isso permitirá a execução de scripts locais não assinados. Após essa configuração, tente ativar o ambiente virtual novamente. Como alternativa, você pode utilizar o Prompt de Comando (`cmd.exe`) ao invés do PowerShell, onde esta restrição não se aplica.

**Problema: Janela do OpenCV não abre ou congela em ambientes Linux sem GUI**

  * *Sintomas:* Em servidores Linux ou ambientes sem interface gráfica (como SSH sem X forwarding), o sistema falha ao tentar abrir a janela de vídeo.
  * *Causa:* O LibraSign foi projetado para ambientes com interface gráfica completa, utilizando janelas do OpenCV para exibição do vídeo.
  * *Solução:* O sistema não é adequado para execução em ambientes sem GUI. Para utilização remota, considere configurar X11 forwarding em sua conexão SSH (`ssh -X` ou `ssh -Y`), ou utilize soluções de desktop remoto como VNC. Alternativamente, o código poderia ser modificado para salvar frames processados em disco ao invés de exibi-los, mas isso está além do escopo do uso padrão do sistema.

**Problema: Desempenho lento, frames travando ou baixa taxa de atualização**

  * *Sintomas:* O vídeo na janela do sistema atualiza de forma muito lenta, apresenta travamentos frequentes, ou a predição demora muito tempo.
  * *Causa:* Hardware insuficiente, outros processos consumindo recursos do sistema, ou câmera de baixa qualidade.
  * *Solução:* Feche outros aplicativos que possam estar consumindo CPU ou RAM significativos. Reduza a resolução da câmera se possível. Em sistemas Linux, considere fechar aplicativos pesados de desktop. Certifique-se de que drivers de vídeo estão atualizados. Como último recurso, considere utilizar um computador com especificações mais robustas.

-----

## Aplicabilidade e Extensibilidade

Embora o LibraSign tenha sido desenvolvido especificamente para o reconhecimento do alfabeto manual da Língua Brasileira de Sinais, sua arquitetura modular e metodologia baseada em landmarks geométricos conferem ao sistema notável flexibilidade e potencial para adaptação a contextos diversos.

  * **Adaptação para Outras Línguas de Sinais:** A estrutura do sistema pode ser prontamente retreinada para reconhecer alfabetos manuais de outras línguas de sinais nacionais, como a American Sign Language (ASL), a British Sign Language (BSL), ou qualquer outra língua de sinais que utilize datilologia. O processo requer apenas a captura de um novo dataset com as configurações de mão específicas da língua-alvo, seguido do retreinamento do modelo conforme descrito neste guia.

  * **Expansão para Vocabulário Mais Amplo:** Pesquisadores interessados em expandir o sistema além do alfabeto manual podem coletar amostras de sinais ideográficos completos (palavras em Libras) e incluí-los como classes adicionais no dataset. Esta expansão demandaria possivelmente arquiteturas de rede mais complexas, capazes de modelar sequências temporais, como Redes Neurais Recorrentes (RNN) ou Transformers, visto que muitos sinais envolvem movimento dinâmico das mãos.

  * **Reconhecimento de Gestos Personalizados:** A metodologia pode ser aplicada para reconhecer conjuntos de gestos personalizados em contextos diversos, como controle gestual de interfaces, interpretação de comandos em ambientes de realidade virtual ou aumentada, ou sistemas de comunicação customizados para necessidades específicas. A versatilidade dos landmarks do MediaPipe permite que praticamente qualquer configuração de mão distinguível seja capturada e classificada.

  * **Integração com Outras Modalidades:** O sistema atual processa exclusivamente a configuração espacial da mão. Trabalhos futuros poderiam integrar informação facial (expressões), corporal (postura e orientação), e contextual (posição no espaço de sinalização) para aproximar-se de um sistema de reconhecimento mais completo da língua de sinais em sua riqueza linguística.

  * **Aplicações Educacionais:** O LibraSign serve como ferramenta didática valiosa para o ensino de conceitos de aprendizado de máquina, visão computacional e processamento de sinais. Estudantes podem experimentar com diferentes arquiteturas de rede, técnicas de pré-processamento, estratégias de aumento de dados e metodologias de validação, utilizando o sistema como plataforma de aprendizado prático.

-----

## Considerações Finais

O LibraSign representa uma contribuição ao campo da tecnologia assistiva e ao reconhecimento automatizado de línguas de sinais, demonstrando a viabilidade de abordagens baseadas em landmarks geométricos para a classificação de gestos manuais. O projeto foi desenvolvido com rigor acadêmico, atenção à reprodutibilidade e compromisso com a disseminação do conhecimento através de código aberto e datasets públicos.

É essencial reiterar que o sistema, em seu estado atual, possui limitações significativas que o posicionam como ferramenta de pesquisa e educação, não como substituto para comunicação profissional ou interpretação da língua de sinais. A Libras, assim como outras línguas de sinais, constitui um sistema linguístico completo e complexo, com gramática, sintaxe, semântica e pragmática próprias que transcendem vastamente a mera soletração manual de letras.

O reconhecimento do alfabeto manual, embora útil em contextos específicos como soletração de nomes próprios ou termos técnicos sem sinal estabelecido, representa apenas uma fração diminuta da comunicação em Libras. A compreensão adequada da língua envolve expressões faciais que modificam significado gramatical, movimento e orientação das mãos no espaço tridimensional de sinalização, uso de classificadores, incorporação e referenciação espacial, entre inúmeros outros elementos linguísticos.

Portanto, enfatiza-se que o LibraSign não deve ser interpretado como sistema de tradução da Libras em sua totalidade, mas sim como um primeiro passo metodológico em direção a sistemas mais abrangentes, e como ferramenta valiosa para o estudo de técnicas de reconhecimento de padrões visuais.

Usuários interessados em aprofundar-se na compreensão técnica do sistema, nos fundamentos teóricos que embasam as decisões arquiteturais, nos resultados experimentais detalhados e nas discussões sobre trabalhos relacionados são encorajados a consultar o documento acadêmico completo referenciado na próxima seção.

-----

## Referências e Documentação Complementar

**📄 Trabalho de Conclusão de Curso Completo:**

  * [HeitorFernandes-TCC\_BSI.pdf](https://github.com/Heitorccf/librasign/blob/master/HeitorFernandes-TCC_BSI.pdf)

Este documento acadêmico apresenta de forma detalhada e rigorosa todos os aspectos do projeto, incluindo:

  * Revisão bibliográfica sobre línguas de sinais, tecnologias assistivas e reconhecimento de gestos.
  * Discussão sobre abordagens metodológicas para processamento de sinais visuais.
  * Fundamentação teórica sobre redes neurais artificiais e perceptrons multicamadas.
  * Descrição detalhada do processo de coleta e preparação do dataset.
  * Análise estatística completa dos resultados experimentais.
  * Matrizes de confusão e métricas de desempenho discriminadas por classe.
  * Discussão sobre limitações do sistema e considerações para trabalhos futuros.
  * Reflexões sobre o impacto social da tecnologia e questões éticas relacionadas.

**Documentação das Bibliotecas Utilizadas:**

Para usuários interessados em compreender mais profundamente as tecnologias empregadas no LibraSign, recomenda-se a consulta da documentação oficial das principais bibliotecas:

  * **MediaPipe:** [https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)
  * **scikit-learn:** [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
  * **OpenCV:** [https://docs.opencv.org/](https://docs.opencv.org/)
  * **NumPy:** [https://numpy.org/doc/](https://numpy.org/doc/)
  * **Pandas:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

**Recursos sobre Libras:**

Para aqueles interessados em aprender mais sobre a Língua Brasileira de Sinais e sua estrutura linguística:

  * **Instituto Nacional de Educação de Surdos (INES):** [http://www.ines.gov.br](http://www.ines.gov.br)
  * **Dicionário de Libras do INES:** Recurso online para consulta de sinais.
  * **Federação Nacional de Educação e Integração dos Surdos (FENEIS):** [https://www.feneis.org.br](https://www.feneis.org.br)

**Contribuições e Feedback:**

O LibraSign é um projeto de código aberto e contribuições da comunidade são bem-vindas. Usuários que identificarem bugs, tiverem sugestões de melhorias ou desejarem contribuir com código são encorajados a abrir issues ou pull requests no repositório do GitHub.

Para questões acadêmicas, dúvidas técnicas ou discussões sobre o projeto, sinta-se à vontade para entrar em contato através dos canais disponibilizados no repositório.

**Agradecimentos:**

O desenvolvimento do LibraSign foi possível graças ao apoio institucional da universidade, à orientação acadêmica recebida, ao acesso a recursos computacionais, e à disponibilização gratuita de bibliotecas de código aberto pela comunidade científica e tecnológica internacional. Agradecimentos especiais à comunidade surda brasileira, cuja língua e cultura inspiram este trabalho e motivam o desenvolvimento de tecnologias mais inclusivas.

**Licença:**

Este projeto é distribuído sob licença de código aberto, permitindo uso, modificação e distribuição de acordo com os termos especificados no arquivo `LICENSE` do repositório. Ao utilizar ou modificar este código, solicita-se que a devida atribuição seja mantida conforme as práticas da comunidade de software livre.

-----

**Última Atualização:** Novembro de 2025

**Autor:** Heitor Fernandes
**Instituição:** Bacharelado em Sistemas de Informação
**Repositório:** [https://github.com/heitorccf/librasign](https://github.com/heitorccf/librasign)
**Dataset Público:** [https://www.kaggle.com/datasets/heitorccf/librasign](https://www.kaggle.com/datasets/heitorccf/librasign)

-----

*Este README foi elaborado com o objetivo de fornecer documentação abrangente e acessível para usuários de diferentes níveis de experiência técnica. Para sugestões de melhorias nesta documentação, por favor, entre em contato através do repositório do GitHub.*