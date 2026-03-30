# 🚦 TrafficIQ — Smart Signal Prediction System

An ML-powered traffic signal wait time predictor for Indian intersections.

---

## 📁 Project Structure

```
traffic-project/
├── backend/
│   ├── app.py                  # Flask API (enhanced)
│   ├── traffic_model.pkl       # Trained ML model
│   ├── model_metadata.json     # Model metrics & feature list
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html              # Dashboard UI
├── model/
│   └── train.py                # Improved training script
└── docker-compose.yml
```

---

## 🚀 Option 1: Run Locally (without Docker)

### Step 1 — Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Start the API
```bash
python app.py
# API runs at http://localhost:5000
```

### Step 3 — Open the dashboard
Open `frontend/index.html` in your browser directly — no server needed.

---

## 🐳 Option 2: Run with Docker Compose (recommended)

```bash
# From the project root
docker-compose up --build
```

- **API:**       http://localhost:5000  
- **Dashboard:** http://localhost:3000  

To stop:
```bash
docker-compose down
```

---

## 🔁 Retrain the Model

```bash
cd model
# Put your CSV here: indian_smart_traffic_dataset_pro.csv
python train.py
# Copies traffic_model.pkl and model_metadata.json → copy to backend/
cp traffic_model.pkl ../backend/
cp model_metadata.json ../backend/
```

---

## 📡 API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/`               | API info                             |
| GET    | `/health`         | Health check                         |
| GET    | `/model-info`     | Model type + accuracy metrics        |
| POST   | `/predict`        | Predict wait time for 1 intersection |
| POST   | `/batch-predict`  | Predict for multiple intersections   |
| GET    | `/simulate`       | Sample predictions for 3 scenarios   |

### POST `/predict` — Example request
```json
{
  "Day": "Monday",
  "Hour": 8,
  "Month": 6,
  "Weather": "Clear",
  "Area_Type": "Commercial",
  "Traffic_Density": "High",
  "Car_Count": 60,
  "Bike_Count": 40,
  "Truck_Count": 10
}
```

### POST `/batch-predict` — Example request
```json
{
  "records": [
    { "Day": "Monday", "Hour": 8, "Weather": "Clear", "Area_Type": "Commercial",
      "Traffic_Density": "High", "Car_Count": 60, "Bike_Count": 40, "Truck_Count": 10 },
    { "Day": "Sunday", "Hour": 2, "Weather": "Clear", "Area_Type": "Highway",
      "Traffic_Density": "Low", "Car_Count": 5, "Bike_Count": 3, "Truck_Count": 1 }
  ]
}
```

---

## 📊 Model Performance

| Metric | Value  |
|--------|--------|
| R²     | 0.83   |
| RMSE   | 18.7 s |
| MAE    | 12.2 s |

**Algorithm:** Gradient Boosting Regressor (sklearn)  
**Key features added:** Time period buckets, total vehicle count, truck ratio, rush-hour × density interaction

---

## 🛠 Tech Stack

- **ML:** scikit-learn (GradientBoostingRegressor)
- **API:** Flask + Flask-CORS, served with Gunicorn
- **Frontend:** Vanilla HTML/CSS/JS
- **Deploy:** Docker + Docker Compose
