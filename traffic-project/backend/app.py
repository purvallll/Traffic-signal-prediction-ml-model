# app.py — Enhanced Traffic Prediction API
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# ── Load model & metadata ─────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "traffic_model.pkl")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.json")

model    = joblib.load(MODEL_PATH)
metadata = json.load(open(METADATA_PATH))
print("✅ Model loaded")

# ── Feature engineering (must match train.py) ─────────────────────────────────
def time_of_day(h):
    if 5  <= h < 10: return "Morning_Rush"
    if 10 <= h < 16: return "Midday"
    if 16 <= h < 20: return "Evening_Rush"
    return "Night"

def enrich(data: dict) -> dict:
    """Add derived features so raw user input matches model input."""
    data = data.copy()
    h = int(data.get("Hour", 0))
    data["Time_Period"]    = time_of_day(h)
    data["Total_Vehicles"] = data["Car_Count"] + data["Bike_Count"] + data["Truck_Count"]
    data["Truck_Ratio"]    = data["Truck_Count"] / (data["Total_Vehicles"] + 1)
    density_map = {"Low": 1, "Medium": 2, "High": 3}
    is_rush = 1 if data["Time_Period"] in ["Morning_Rush", "Evening_Rush"] else 0
    data["Rush_x_Density"] = is_rush * density_map.get(data.get("Traffic_Density", "Low"), 1)
    data.setdefault("Month", datetime.now().month)
    data.setdefault("Is_Weekend", 1 if data.get("Day") in ["Saturday", "Sunday"] else 0)
    return data

def signal_advice(seconds: float) -> dict:
    if seconds < 20:
        return {"level": "Low",    "color": "green",  "message": "Light traffic — short wait expected."}
    if seconds < 60:
        return {"level": "Medium", "color": "orange", "message": "Moderate traffic — plan for a short delay."}
    return {"level": "High",   "color": "red",    "message": "Heavy traffic — consider an alternate route."}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "message": "🚦 Traffic Prediction API",
        "version": "2.0",
        "endpoints": ["/predict", "/batch-predict", "/model-info", "/health"]
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/model-info")
def model_info():
    return jsonify({
        "model_type": metadata["model_type"],
        "metrics":    metadata["metrics"],
        "features":   metadata["features"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict signal waiting time for a single intersection.
    Required fields: Day, Hour, Weather, Area_Type, Traffic_Density,
                     Car_Count, Bike_Count, Truck_Count
    """
    try:
        data = request.get_json(force=True)
        enriched = enrich(data)
        df = pd.DataFrame([enriched])[metadata["features"]]
        pred = float(model.predict(df)[0])
        pred = max(5, round(pred, 1))

        return jsonify({
            "status": "success",
            "predicted_waiting_time_seconds": pred,
            "advice": signal_advice(pred),
            "input_used": enriched
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """
    Predict for multiple intersections at once.
    Body: { "records": [ {...}, {...} ] }
    """
    try:
        records = request.get_json(force=True).get("records", [])
        if not records:
            return jsonify({"status": "error", "error": "No records provided"}), 400

        results = []
        for rec in records:
            enriched = enrich(rec)
            df = pd.DataFrame([enriched])[metadata["features"]]
            pred = float(model.predict(df)[0])
            pred = max(5, round(pred, 1))
            results.append({
                "predicted_waiting_time_seconds": pred,
                "advice": signal_advice(pred)
            })

        avg = round(np.mean([r["predicted_waiting_time_seconds"] for r in results]), 1)
        return jsonify({
            "status": "success",
            "count": len(results),
            "average_waiting_time_seconds": avg,
            "predictions": results
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

@app.route("/simulate", methods=["GET"])
def simulate():
    """Return sample predictions for different conditions (demo use)."""
    scenarios = [
        {"Day": "Monday",   "Hour": 8,  "Weather": "Clear", "Area_Type": "Commercial",  "Traffic_Density": "High",   "Car_Count": 60, "Bike_Count": 40, "Truck_Count": 10},
        {"Day": "Saturday", "Hour": 14, "Weather": "Rainy", "Area_Type": "Residential", "Traffic_Density": "Medium", "Car_Count": 25, "Bike_Count": 20, "Truck_Count": 3},
        {"Day": "Sunday",   "Hour": 2,  "Weather": "Clear", "Area_Type": "Highway",     "Traffic_Density": "Low",    "Car_Count": 5,  "Bike_Count": 3,  "Truck_Count": 1},
    ]
    results = []
    for s in scenarios:
        enriched = enrich(s)
        df = pd.DataFrame([enriched])[metadata["features"]]
        pred = max(5, round(float(model.predict(df)[0]), 1))
        results.append({**s, "predicted_waiting_time_seconds": pred, "advice": signal_advice(pred)})
    return jsonify({"status": "success", "scenarios": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
