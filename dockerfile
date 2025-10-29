# Usa una imagen oficial de Python ligera
FROM python:3.10-slim

# Instala dependencias del sistema necesarias para OpenCV, EasyOCR y YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crea y activa el entorno
WORKDIR /app
COPY . /app

# Instala dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto estándar de Railway
EXPOSE 8000

# Comando para arrancar FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
