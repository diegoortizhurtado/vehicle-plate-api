from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
from ultralytics import YOLO
import easyocr
from io import BytesIO

# ---- CONFIG ----
CONFIG_PATH = "config.json"
CLASSES_PATH = "classes.json"
MODEL_PATH = "model_vehicle.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- CARGAR CONFIG ----
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
with open(CLASSES_PATH, 'r') as f:
    class_names = json.load(f)

IMG_SIZE = config["img_size"]
mean = config["mean"]
std = config["std"]

# ---- MODELO CNN ----
class ModelCNN(nn.Module):
    def __init__(self, num_classes):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---- CARGAR MODELO VEHÍCULO ----
vehicle_model = ModelCNN(config["num_classes"]).to(DEVICE)
vehicle_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
vehicle_model.eval()

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ---- CARGAR YOLO (descarga automática) ----
print("Cargando detector de placas YOLOv8...")
plate_model = YOLO("keremberke/yolov8n-license-plate")

# ---- OCR ----
ocr = easyocr.Reader(['en'], gpu=False)

app = FastAPI(title="Vehicle and License Plate Recognition API")

# ---- FUNCIONES ----
def read_image(file: UploadFile):
    image = Image.open(BytesIO(file.file.read())).convert("RGB")
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_image(file)
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # --- Clasificación carro/moto ---
    with torch.no_grad():
        outputs = vehicle_model(tensor)
        _, pred = torch.max(outputs, 1)
        pred_class = class_names[pred.item()]
        confidence = torch.softmax(outputs, dim=1)[0][pred.item()].item()

    # --- Detección de placas ---
    plate_data = []
    results = plate_model.predict(np.array(image), verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        crop = np.array(image)[y1:y2, x1:x2]
        text = ocr.readtext(crop, detail=0)
        if text:
            plate_data.append({
                "bbox": [x1, y1, x2, y2],
                "plate_text": text[0]
            })

    response = {
        "vehicle_type": pred_class,
        "confidence": round(float(confidence), 4),
        "plates_detected": plate_data
    }

    return JSONResponse(content=response)
