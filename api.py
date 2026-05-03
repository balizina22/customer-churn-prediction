from fastapi import FastAPI
import pickle
import json
import numpy as np
from utils import build_input_from_dict

app = FastAPI()

# Chargement
with open("models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)

THRESHOLD = 0.40


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    try:
        df = build_input_from_dict(data, feature_names)
        scaled = scaler.transform(df)

        proba = model.predict_proba(scaled)[0][1]
        prediction = int(proba >= THRESHOLD)

        return {
            "probability": float(proba),
            "prediction": prediction
        }

    except Exception as e:
        return {"error": str(e)}