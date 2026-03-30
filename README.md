# Traffic-signal-prediction-ml-model
# 🚦 TrafficPoce

### Predicting urban traffic signal wait times using machine learning

TrafficPoce is a full-stack ML web application that predicts how long vehicles will wait at traffic signals across Indian cities, factoring in weather, area type, time of day, vehicle mix, and traffic density. Built end-to-end from data to dashboard.

---

## ✨ What it does

You give it an intersection's conditions. It tells you how long you'll wait — and whether to rethink your route.

- 🟢 **Under 20s** — light traffic, smooth flow
- 🟠 **20–60s** — moderate congestion, minor delay
- 🔴 **60s+** — heavy traffic, consider alternatives

The dashboard updates in real time, keeps a prediction history, and visualizes wait intensity on a live gauge.

---

## 🧠 The Model

Trained on **15,000 rows** of real Indian traffic data using a **Gradient Boosting Regressor** — a step up from the baseline Random Forest, with custom feature engineering on top.

Features engineered from raw data:
- Time-of-day buckets — Morning Rush, Midday, Evening Rush, Night
- Total vehicle volume & truck-to-vehicle ratio
- Rush hour × traffic density interaction term

| Metric | Score |
|--------|-------|
| R² | **0.83** |
| RMSE | 18.7 seconds |
| MAE | 12.2 seconds |

---

## 🗂️ Project Structure

```
traffic-project/
├── backend/
│   ├── app.py                # Flask REST API (5 endpoints)
│   ├── traffic_model.pkl     # Trained model
│   ├── model_metadata.json   # Metrics + feature registry
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html            # Single-page dashboard
├── model/
│   ├── train.py              # Training pipeline
│   └── indian_smart_traffic_dataset_pro.csv
└── docker-compose.yml
```

---

## 🚀 Running Locally

You'll need **Python 3.12** installed.

**1. Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**2. Start the API**
```bash
python app.py
# → http://127.0.0.1:5000
```

**3. Serve the dashboard** (new terminal)
```bash
cd frontend
python -m http.server 3000
# → http://127.0.0.1:3000
```

Open your browser at **http://127.0.0.1:3000** and start predicting.

---

## 🐳 Or just use Docker

```bash
docker-compose up --build
```

That's it. API at `:5000`, dashboard at `:3000`.

---

## 📡 API Endpoints

| Method | Route | What it does |
|--------|-------|--------------|
| `GET` | `/health` | Ping the server |
| `GET` | `/model-info` | Model type, metrics, feature list |
| `POST` | `/predict` | Single intersection prediction |
| `POST` | `/batch-predict` | Multiple intersections at once |
| `GET` | `/simulate` | 3 pre-built scenario predictions |

**Sample request to `/predict`:**
```json
{
  "Day": "Monday",
  "Hour": 8,
  "Weather": "Clear",
  "Area_Type": "Commercial",
  "Traffic_Density": "High",
  "Car_Count": 60,
  "Bike_Count": 40,
  "Truck_Count": 10
}
```

**Response:**
```json
{
  "predicted_waiting_time_seconds": 87.4,
  "advice": {
    "level": "High",
    "color": "red",
    "message": "Heavy traffic — consider an alternate route."
  }
}
```

---

## 🔁 Retraining the Model

Drop in new data and retrain anytime:

```bash
cd model
python train.py
cp traffic_model.pkl ../backend/
cp model_metadata.json ../backend/
```

---

## 🛠️ Stack

- **ML** — scikit-learn, pandas, numpy
- **API** — Flask, Flask-CORS, Gunicorn
- **Frontend** — HTML, CSS, vanilla JavaScript
- **Infra** — Docker, Docker Compose

---

> Built with by Purval <3
