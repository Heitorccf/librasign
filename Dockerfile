# Etapa 1: Definir a imagem base
# Usaremos uma imagem oficial do Python 3.9. A versão "slim" é mais leve.
FROM python:3.9-slim

# Etapa 2: Instalar dependências do sistema operacional
# OpenCV precisa de algumas bibliotecas do sistema para funcionar corretamente,
# especialmente para processamento de vídeo e para exibir a janela (GUI).
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# Etapa 3: Configurar o ambiente de trabalho dentro do contêiner
WORKDIR /app

# Etapa 4: Instalar as dependências do Python
# Copiamos primeiro o requirements.txt para aproveitar o cache do Docker.
# Se este arquivo não mudar, o Docker não reinstalará tudo a cada build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Etapa 5: Copiar o código da aplicação e os modelos
COPY src/ ./src/
COPY models/ ./models/

# Etapa 6: Comando para executar a aplicação
# Este comando será executado quando o contêiner iniciar.
CMD ["python", "src/predict.py"]