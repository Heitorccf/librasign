# 🖐️ LibraSign

## ℹ️ Introdução

O LibraSign é um sistema de código aberto desenvolvido como Trabalho de Conclusão de Curso que utiliza técnicas de visão computacional e aprendizado de máquina para reconhecer em tempo real os gestos correspondentes ao alfabeto manual da Língua Brasileira de Sinais. O projeto explora metodologias de processamento de dados geométricos e classificação neural para aplicações de acessibilidade comunicacional.

O sistema reconhece exclusivamente as configurações de mão correspondentes às letras do alfabeto manual, de A a Z. Esta delimitação foi estabelecida para permitir uma investigação acadêmica focada na eficácia de redes neurais artificiais na classificação de gestos estáticos. O projeto destina-se primariamente ao ambiente acadêmico e educacional, não substituindo intérpretes profissionais ou servindo para uso comunicacional cotidiano em larga escala.

---

## 📚 Fundamentação Acadêmica

A fundamentação teórica completa, incluindo revisão de literatura sobre línguas de sinais, técnicas de visão computacional, arquiteturas de redes neurais, metodologia experimental, análise estatística dos resultados e discussão sobre as implicações sociais da tecnologia assistiva, encontra-se detalhada no documento acadêmico completo disponível neste repositório: **[HeitorFernandes-TCC_BSI.pdf](https://github.com/Heitorccf/librasign/blob/master/HeitorFernandes-TCC_BSI.pdf)**.

O documento aborda a diferenciação entre a comunicação em línguas de sinais e a datilologia, as limitações das abordagens baseadas em processamento de imagens brutas, a escolha por representações geométricas de landmarks e as métricas de desempenho obtidas através de validação cruzada estratificada.

---

## 🔭 Visão Geral do Sistema

### 🚧 Escopo e Limitações

O sistema foi desenvolvido especificamente para reconhecer as configurações de mão estáticas do alfabeto manual da Libras. Esta escolha metodológica foi deliberada e alinha-se com os objetivos de pesquisa do projeto.

O sistema reconhece as configurações de mão correspondentes a cada uma das letras de A a Z quando apresentadas de forma estática e isolada diante da câmera. Não reconhece palavras completas em Libras, sinais compostos ou ideográficos, expressões faciais, movimento corporal, utilização do espaço de sinalização, variações regionais ou transições dinâmicas entre letras.

---

## 🏗️ Arquitetura do Sistema

A arquitetura compreende três módulos principais:

1.  **Módulo de Captura:** Utiliza a biblioteca MediaPipe do Google para acessar a câmera e realizar a detecção em tempo real das mãos. Para cada frame capturado, o MediaPipe identifica vinte e um pontos de referência anatômicos na mão detectada, extraindo suas coordenadas tridimensionais no espaço normalizado. Estes dados geométricos são persistidos em arquivos CSV organizados por classe.

2.  **Módulo de Treinamento:** Implementa o pipeline completo de aprendizado supervisionado. Após carregar o dataset de landmarks, aplica transformação de normalização geométrica que torna os dados invariantes à posição absoluta da mão e à escala. Os dados são então padronizados utilizando `StandardScaler`. O modelo escolhido é um Perceptron Multicamadas com duas camadas ocultas contendo 128 e 64 neurônios, treinado através do algoritmo de retropropagação. A avaliação do desempenho é conduzida através de validação cruzada estratificada com cinco partições.

3.  **Módulo de Predição:** Carrega os artefatos persistidos, inicializa a captura de vídeo e processa cada frame. Para cada detecção de mão, as coordenadas dos landmarks são extraídas, normalizadas e padronizadas exatamente da mesma forma que durante o treinamento. O sistema implementa filtro de votação majoritária sobre os últimos dez frames e mecanismo de confirmação temporal que exige que uma letra permaneça estável por dois segundos antes de ser adicionada à frase em construção.

---

## 🔄 Fluxo de Processamento

O sistema captura continuamente frames da câmera do dispositivo. Cada frame é processado pelo modelo de detecção de mãos do MediaPipe, que identifica a presença e localização de mãos na imagem. O MediaPipe identifica vinte e um pontos anatômicos na mão detectada, cada landmark representado por suas coordenadas tridimensionais.

Os landmarks brutos são transformados através de normalização geométrica, com todos os pontos transladados para que o pulso fique na origem e escalonados pela distância entre o pulso e a base do dedo médio. Os dados normalizados são então padronizados utilizando o `StandardScaler` treinado. O vetor de características padronizado é propagado através das camadas do Perceptron Multicamadas, com a camada de saída produzindo probabilidades para cada classe.

Para reduzir oscilações, o sistema aplica filtro de votação majoritária sobre as últimas dez predições. Uma letra só é confirmada se permanecer como predição predominante por dois segundos consecutivos. O sistema renderiza sobre o vídeo os landmarks detectados, a letra reconhecida, uma barra de progresso para confirmação e, na parte inferior, a sentença formada pelas letras confirmadas.

---

## 💻 Requisitos do Sistema

### 💿 Requisitos de Software

O sistema foi desenvolvido e testado em ambientes Linux, macOS e Windows. É necessária a instalação do Python na versão **3.11.13**. Versões anteriores à 3.9 não são suportadas. O sistema requer acesso a uma câmera funcional para captura de vídeo em tempo real.

As bibliotecas essenciais e suas versões são:
* `scikit-learn==1.7.2`
* `numpy==2.2.6`
* `pandas==2.3.2`
* `opencv-python==4.12.0.88`
* `mediapipe==0.10.14`
* `kagglehub==0.3.13`

### 🖥️ Requisitos de Hardware

Recomenda-se um processador com pelo menos dois núcleos físicos operando a dois gigahertz ou superior. Um mínimo de quatro gigabytes de memória RAM é necessário, sendo recomendados oito gigabytes ou mais. O projeto e o dataset público ocupam menos de cem megabytes, recomendando-se ter pelo menos um gigabyte de espaço livre.

A câmera deve ter resolução mínima de 640x480 pixels e taxa de captura de pelo menos 15 frames por segundo. Câmeras com resolução HD ou superior com 30 frames por segundo proporcionam melhor experiência. Condições adequadas de iluminação são cruciais, recomendando-se ambiente bem iluminado, evitando contraluz intenso ou sombras fortes.

---

## 📥 Guia de Instalação

### 1️⃣ Preparação do Ambiente

Verifique a instalação do Python executando no terminal o comando:

```bash
python --version
````

Ou:

```bash
python3 --version
```

O comando deve retornar uma versão 3.11.x. Se o comando não for reconhecido ou a versão for inferior, instale ou atualize o Python seguindo as instruções específicas para seu sistema operacional.

No Linux Debian ou Ubuntu, utilize:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

No macOS com Homebrew, execute:

```bash
brew install python@3.11
```

No Windows, baixe o instalador oficial do python.org, marcando a opção de adicionar o Python ao PATH durante a instalação.

### 2️⃣ Clonagem do Repositório

Com o Python instalado, obtenha uma cópia local do repositório. Certifique-se de ter o Git instalado verificando com:

```bash
git --version
```

Navegue até o diretório onde deseja armazenar o projeto e execute:

```bash
git clone [https://github.com/heitorccf/librasign.git](https://github.com/heitorccf/librasign.git)
```

Entre no diretório recém-criado com:

```bash
cd librasign
```

Todos os comandos subsequentes devem ser executados a partir desta pasta.

### 3️⃣ Configuração do Ambiente Virtual

O uso de ambiente virtual é recomendado para isolar as dependências do projeto.

No Linux e macOS, execute:

```bash
python3 -m venv .venv
```

No Windows, execute:

```bash
python -m venv .venv
```

Para ativar o ambiente virtual, no Linux e macOS execute:

```bash
source .venv/bin/activate
```

No Windows com Prompt de Comando, execute:

```bash
.venv\Scripts\activate.bat
```

No Windows com PowerShell, execute:

```bash
.venv\Scripts\Activate.ps1
```

No PowerShell, talvez seja necessário ajustar a política de execução com:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Após a ativação, seu terminal deve exibir o prefixo `(.venv)` no início do prompt.

### 4️⃣ Instalação das Dependências

Com o ambiente virtual ativado, atualize o pip com:

```bash
python -m pip install --upgrade pip
```

Instale as bibliotecas listadas no `requirements.txt` executando:

```bash
pip install -r requirements.txt
```

O pip instalará todas as bibliotecas necessárias, incluindo opencv-python, mediapipe, scikit-learn e kagglehub.

Teste se as bibliotecas principais foram instaladas executando:

```bash
python -c "import cv2, mediapipe, sklearn, numpy, pandas; print('Todas as bibliotecas foram importadas com sucesso')"
```

Se a mensagem de sucesso for exibida, o ambiente está pronto.

-----

## 🚀 Execução do Sistema

### ▶️ Modo de Uso Padrão

Este modo utiliza o modelo pré-treinado disponível na pasta models para reconhecimento em tempo real. Com o ambiente virtual ativado, execute:

```bash
python src/predict.py
```

O sistema carregará o modelo e abrirá uma janela gráfica exibindo o vídeo da câmera.

Posicione sua mão no campo de visão da câmera contra um fundo relativamente uniforme. Forme a configuração de mão correspondente a uma letra do alfabeto manual e mantenha a posição estável. Uma barra de progresso verde indicará o tempo para confirmação do gesto. Após dois segundos, a letra será confirmada e adicionada à frase na parte inferior da tela.

Os controles do teclado são:

  * Tecla **ESC** para encerrar a aplicação.
  * **Backspace** para remover a última letra adicionada à frase.
  * Tecla **C** para limpar completamente a frase.

### 📷 Captura de Novo Dataset

Usuários que desejam capturar seus próprios dados podem usar o script de captura executando:

```bash
python src/capture.py
```

O sistema abrirá uma janela de vídeo e aguardará comandos.

Pressione a tecla da letra que deseja capturar. A captura iniciará automaticamente quando uma mão for detectada. Forme o gesto da letra escolhida e mova levemente a mão para criar variabilidade nos dados. O sistema capturará até mil amostras por letra, salvando os landmarks no diretório `data/landmarks` em arquivos CSV.

Os controles durante a captura são:

  * Teclas **A a Z** para iniciar ou alternar a captura.
  * Tecla **zero** para capturar amostras da classe "nenhum".
  * Tecla **espaço** para pausar ou retomar a captura.
  * Tecla **ESC** para encerrar o script.

### ⚙️ Retreinamento do Modelo

Após capturar um dataset personalizado, execute:

```bash
python src/train.py
```

O script baixará o dataset público de referência do Kaggle, carregará todos os arquivos CSV do diretório de landmarks, aplicará a normalização geométrica e a padronização, executará a validação cruzada estratificada com cinco partições para avaliar o modelo, exibirá a acurácia média e o desvio padrão, treinará um modelo final usando todos os dados e salvará os novos artefatos no diretório `models`.

-----

## 📊 Dataset Público

O dataset de landmarks usado no desenvolvimento do LibraSign foi disponibilizado publicamente na plataforma Kaggle. Ele contém aproximadamente mil amostras para cada uma das vinte e sete classes, totalizando cerca de vinte e sete mil exemplos. O dataset pode ser acessado através do link:

🔗 [https://www.kaggle.com/datasets/heitorccf/librasign](https://www.kaggle.com/datasets/heitorccf/librasign)

O dataset consiste em arquivos CSV, um para cada classe, onde cada linha representa uma amostra contendo sessenta e três valores numéricos correspondentes às coordenadas x, y e z dos vinte e um landmarks da mão. Pesquisadores e desenvolvedores podem utilizar este dataset para reproduzir os resultados apresentados no trabalho, explorar diferentes arquiteturas de redes neurais, desenvolver outros sistemas de reconhecimento de gestos baseados em landmarks ou expandir o sistema com classes adicionais.

-----

## 🛠️ Solução de Problemas

  * Se o comando `python` não for reconhecido ou exibir versão 2.x, use `python3` em vez de python para todos os comandos.

  * Se o sistema falhar ao acessar a webcam, verifique as permissões de privacidade do sistema operacional para permitir acesso à câmera.

  * Se o Python não encontrar bibliotecas como `cv2` ou `mediapipe`, verifique se o ambiente virtual está ativado observando o prefixo `(.venv)` no terminal. Se não estiver ativado, execute o comando de ativação apropriado e reinstale as dependências com:

<!-- end list -->

```bash
pip install -r requirements.txt
```

  * Se os landmarks da mão não aparecerem ou piscarem na tela, melhore a iluminação do ambiente, use um fundo simples e de cor uniforme e ajuste a distância da mão para a câmera entre trinta e sessenta centímetros.

  * Se o sistema confundir letras frequentemente, revise o dataset personalizado garantindo que os gestos estão corretos e capture mais amostras com variabilidade. A confusão entre pares geometricamente semelhantes como M e N, G e Q, ou F e T é uma limitação conhecida do modelo atual.

-----

## 🌍 Aplicabilidade e Extensibilidade

Embora o LibraSign seja focado no alfabeto manual da Libras, sua arquitetura baseada em landmarks geométricos oferece flexibilidade para adaptação. A estrutura pode ser retreinada para reconhecer alfabetos manuais de outras línguas de sinais. O processo requer apenas a captura de um novo dataset com as configurações de mão da língua-alvo e o retreinamento do modelo.

O sistema pode ser expandido para reconhecer sinais ideográficos, o que exigiria a coleta de amostras desses sinais e, possivelmente, a mudança para arquiteturas de rede capazes de modelar sequências temporais, já que muitos sinais envolvem movimento dinâmico. A metodologia pode ser aplicada para reconhecer conjuntos de gestos personalizados para outros fins, como controle de interfaces ou comandos em realidade virtual. O LibraSign serve também como ferramenta didática para o ensino de conceitos de aprendizado de máquina, visão computacional e processamento de sinais.

-----

## 📝 Considerações Finais

O LibraSign demonstra a viabilidade de abordagens baseadas em landmarks geométricos para a classificação de gestos manuais. O projeto foi desenvolvido com atenção à reprodutibilidade, disponibilizando o código e o dataset publicamente.

É essencial reiterar que o sistema, em seu estado atual, possui limitações que o posicionam como ferramenta de pesquisa e educação, não como substituto para interpretação profissional. A Libras é um sistema linguístico completo que transcende a soletração manual, envolvendo gramática espacial, expressões faciais e movimento corporal. O reconhecimento do alfabeto manual representa apenas uma pequena fração da comunicação em Libras.

Portanto, o LibraSign deve ser visto como um primeiro passo metodológico em direção a sistemas mais completos e como ferramenta para o estudo de técnicas de reconhecimento de padrões visuais. Usuários interessados no entendimento técnico aprofundado do sistema são encorajados a consultar o documento acadêmico completo.

-----

## 🔗 Referências e Documentação Complementar

O trabalho de conclusão de curso completo está disponível no arquivo **[HeitorFernandes-TCC\_BSI.pdf](https://github.com/Heitorccf/librasign/blob/master/HeitorFernandes-TCC_BSI.pdf)**. Este documento apresenta de forma detalhada todos os aspectos do projeto, incluindo revisão bibliográfica sobre línguas de sinais e tecnologias assistivas, discussão sobre abordagens metodológicas para processamento de sinais visuais, fundamentação teórica sobre redes neurais, descrição do processo de coleta e preparação do dataset, análise estatística dos resultados experimentais e discussão sobre limitações do sistema e trabalhos futuros.

Para usuários interessados em compreender mais profundamente as tecnologias empregadas, recomenda-se a consulta da documentação oficial das principais bibliotecas: MediaPipe, scikit-learn, OpenCV, NumPy e Pandas. Para aqueles interessados em aprender mais sobre a Língua Brasileira de Sinais, sugere-se consultar o Instituto Nacional de Educação de Surdos e a Federação Nacional de Educação e Integração dos Surdos.

O LibraSign é um projeto de código aberto e contribuições da comunidade são bem-vindas através do repositório no GitHub. O desenvolvimento foi possível graças ao apoio institucional, à orientação acadêmica e à disponibilização gratuita de bibliotecas de código aberto pela comunidade. Este projeto é distribuído sob a licença GNU General Public License v3.0.

<br>

**Repositório:** [https://github.com/heitorccf/librasign](https://github.com/heitorccf/librasign)

**Dataset Público:** [https://www.kaggle.com/datasets/heitorccf/librasign](https://www.kaggle.com/datasets/heitorccf/librasign)

**Autor:** Heitor Câmara Costa Fernandes

**Instituição:** Instituto Federal de Educação, Ciência e Tecnologia de São Paulo (IFSP)

**Curso:** Bacharelado em Sistemas de Informação

**Contato:** [heitorccfernandes550@gmail.com](mailto:heitorccfernandes550@gmail.com) | [heitorccf2004@gmail.com](mailto:heitorccf2004@gmail.com)

**Última Atualização:** Novembro de 2025
