# Vehicle & License Plate Recognition API

Este servicio combina tu CNN personalizada con un detector de placas YOLOv8 y OCR EasyOCR.

## 🚀 Instrucciones para Render

1. Crea un nuevo repositorio en GitHub con estos archivos.
2. Sube tus archivos:
   - `model_vehicle.pt`
   - `config.json`
   - `classes.json`
3. En Render:
   - **New Web Service**
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Espera que inicie. Verás logs como: