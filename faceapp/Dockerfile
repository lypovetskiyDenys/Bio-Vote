FROM python:3.9-slim

WORKDIR /app

# Установка необходимых зависимостей для OpenCV и других компонентов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
COPY requirements.txt .
COPY app.py .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт для сервиса
EXPOSE 8000

# Запуск сервиса
CMD ["python", "app.py"]