FROM python:3.11.13-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN mkdir -p /app/models

CMD ["python", "src/predict.py"]