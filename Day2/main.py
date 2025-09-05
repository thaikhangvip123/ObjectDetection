from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os, shutil
from ultralytics import YOLO
import uuid
import glob

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODEL_PATH = "C:\\Users\\thaik\\runs\\detect\\train23\\weights\\best.pt"  # change to your model path

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    # Save uploaded file
    file_ext = os.path.splitext(file.filename)[-1]
    temp_filename = f"{uuid.uuid4().hex}{file_ext}"
    input_path = os.path.join("uploads", temp_filename)
    
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Predict with YOLOv8 and save the result in outputs/
    results = model.predict(source=input_path, save=True, project="outputs", name="result", exist_ok=True)

    # Find the most recent image in outputs/result/
    result_images = glob.glob("outputs/result/*.jpg") + glob.glob("outputs/result/*.png")
    if not result_images:
        return {"error": "No result image generated."}

    result_images.sort(key=os.path.getmtime, reverse=True)  # Sort by last modified
    output_path = result_images[0]

    return FileResponse(output_path, media_type="image/jpeg")