from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/shortfall_xgb.pkl")

class OperationalInput(BaseModel):
    rainfall_mm: float
    soil_moisture_pct: float
    active_shovels: int

@app.get("/")
def serve_home():
    return FileResponse("templates/index.html")

@app.get("/api/anomalies")
def get_anomalies():
    if os.path.exists("data/anomalies.json"):
        with open("data/anomalies.json") as f:
            return json.load(f)
    return []

@app.post("/api/predict-shortfall")
def predict_shortfall(data: OperationalInput):
    inputs = np.array([[data.rainfall_mm, data.soil_moisture_pct, data.active_shovels, data.active_shovels * 2, 1]])
    pred = float(model.predict(inputs)[0])
    pred = max(0.0, round(pred, 2))
    risk = "HIGH" if pred > 400 else "MODERATE" if pred > 150 else "LOW"
    action = "Reroute haul trucks to Bench-4; activate pumps" if risk == "HIGH" else "Normal operations"
    return {"predicted_shortfall_tons": pred, "risk_level": risk, "action": action}