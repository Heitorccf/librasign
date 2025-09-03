<<<<<<< HEAD
=======
Sistema de Tradução em Tempo Real do Alfabeto da Libras (Librasign)

!(https://img.shields.io/badge/status-Concluído-success.svg)

Português | 

Contato

Heitor Câmara Costa Fernandes

🇧🇷 Português

Metodologia e Funcionamento ⚙️

O sistema opera através de um pipeline de processamento que se divide em três etapas fundamentais: coleta de dados, treinamento do modelo e tradução em tempo real.

1. Coleta e Representação de Dados Geométricos 🖐️

Em vez de capturar e armazenar milhares de imagens, o sistema utiliza uma webcam para detectar a mão do usuário em tempo real. A biblioteca MediaPipe é empregada para identificar e extrair as coordenadas 3D de 21 pontos de referência (landmarks) que compõem a estrutura da mão.  Cada gesto é, portanto, convertido em um vetor numérico de 63 dimensões (21 pontos x 3 coordenadas), representando sua geometria esquelética. Esses vetores são salvos em arquivos 

.csv, criando um dataset leve, preciso e imune a ruídos visuais como variações de iluminação ou complexidade do fundo.

2. Treinamento do Modelo de Classificação 🧠

Com os dados geométricos coletados, um modelo de machine learning é treinado para associar cada vetor de coordenadas a uma letra do alfabeto. Devido à natureza tabular e de baixa dimensionalidade dos dados, optou-se por um modelo Perceptron de Múltiplas Camadas (MLP), uma arquitetura de rede neural eficiente para este tipo de tarefa. Antes do treinamento, os dados passam por um processo de normalização (standardization) para garantir que todas as características contribuam de forma equitativa para o aprendizado. O resultado é um modelo classificador altamente otimizado, treinado em segundos, que aprende a distinguir os gestos unicamente a partir de sua forma e estrutura.

3. Tradução em Tempo Real 🚀

A aplicação final integra os componentes anteriores para fornecer uma tradução instantânea. O sistema captura o vídeo da webcam, extrai os landmarks da mão em cada quadro, aplica a mesma normalização utilizada no treinamento e alimenta o vetor de coordenadas ao modelo MLP treinado. O modelo, então, prediz a qual letra o gesto corresponde, e o resultado é exibido na tela para o usuário. Este ciclo de detecção, processamento e classificação ocorre de forma contínua e com baixa latência, criando uma ferramenta de comunicação interativa e funcional.

Publicação Acadêmica 🎓

O documento completo do Trabalho de Conclusão de Curso, contendo a fundamentação teórica, a metodologia detalhada e a análise dos resultados, está disponível para visualização e download em repositórios acadêmicos permanentes.

    Zenodo (DOI): ``

    ResearchGate: ``

Tecnologias Utilizadas 🛠️

    Linguagem: Python 3.9+

    Visão Computacional: OpenCV, MediaPipe

    Machine Learning: Scikit-learn (MLPClassifier, StandardScaler)

    Manipulação de Dados: NumPy, Pandas

Licença ©️

Este projeto está licenciado sob a GNU General Public License v3.0. Veja o arquivo LICENSE para mais detalhes.

Agradecimentos 🙏

    Ao Professor Orientador Cecílio Merlotti Rodas, pelo suporte e direcionamento acadêmico.

    Ao Instituto Federal de Educação, Ciência e Tecnologia de São Paulo (IFSP), pela estrutura e fomento à pesquisa.

Este projeto foi desenvolvido com o objetivo de promover a inclusão e a acessibilidade através da tecnologia. Que ele possa contribuir para quebrar barreiras de comunicação e aproximar pessoas.

🇬🇧 🇺🇸 English

Methodology and How It Works

The system operates through a processing pipeline divided into three fundamental stages: data collection, model training, and real-time translation.

1. Geometric Data Collection and Representation

Instead of capturing and storing thousands of images, the system uses a webcam to detect the user's hand in real-time. The MediaPipe library is employed to identify and extract the 3D coordinates of 21 reference points (landmarks) that make up the hand's structure.  Each gesture is thus converted into a 63-dimensional numerical vector (21 points x 3 coordinates), representing its skeletal geometry. These vectors are saved into 

.csv files, creating a lightweight, precise dataset that is immune to visual noise such as lighting variations or background complexity.

2. Classification Model Training

With the geometric data collected, a machine learning model is trained to associate each coordinate vector with a letter of the alphabet. Due to the tabular and low-dimensional nature of the data, a Multi-Layer Perceptron (MLP) model was chosen, an efficient neural network architecture for this type of task. Before training, the data undergoes a standardization process to ensure that all features contribute equally to the learning process. The result is a highly optimized classifier model, trained in seconds, that learns to distinguish gestures solely based on their shape and structure.

3. Real-Time Translation

The final application integrates the previous components to provide instantaneous translation. The system captures video from the webcam, extracts the hand landmarks in each frame, applies the same normalization used during training, and feeds the coordinate vector to the trained MLP model. The model then predicts which letter the gesture corresponds to, and the result is displayed on the screen for the user. This cycle of detection, processing, and classification occurs continuously and with low latency, creating an interactive and functional communication tool.

Academic Publication

The full Final Year Project document, containing the theoretical foundation, detailed methodology, and analysis of the results, is available for viewing and download in permanent academic repositories.

    Zenodo (DOI): ``

    ResearchGate: ``

Technology Stack

    Linguagem: Python 3.9+

    Visão Computacional: OpenCV, MediaPipe

    Machine Learning: Scikit-learn (MLPClassifier, StandardScaler)

    Manipulação de Dados: NumPy, Pandas

License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.

Acknowledgments

    To my advisor, Professor Cecílio Merlotti Rodas, for the academic support and guidance.

    To the Federal Institute of Education, Science and Technology of São Paulo (IFSP), for the infrastructure and encouragement of research.

This project was developed with the goal of promoting inclusion and accessibility through technology. May it contribute to breaking down communication barriers and bringing people closer together.
>>>>>>> d5eb1054a661822aafbe54037522813806705410
