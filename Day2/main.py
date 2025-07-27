from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # or pickle
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model.joblib")  # You can also use pickle

# Define input format
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.feature1, data.feature2, data.feature3]])
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
