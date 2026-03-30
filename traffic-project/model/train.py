# train.py — Improved Traffic Prediction Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
import json

DATA_FILE = "indian_smart_traffic_dataset_pro.csv"

if not os.path.exists(DATA_FILE):
    print(f"❌ Dataset not found: {DATA_FILE}")
    exit()

# ── 1. Load ──────────────────────────────────────────────────────────────────
data = pd.read_csv(DATA_FILE)
print(f"✅ Loaded {len(data):,} rows")

# ── 2. Feature Engineering ───────────────────────────────────────────────────
data["Timestamp"] = pd.to_datetime(data["Timestamp"])
data["Hour"]       = data["Timestamp"].dt.hour
data["Month"]      = data["Timestamp"].dt.month
data["Is_Weekend"] = data["Timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

# Time-of-day buckets
def time_of_day(h):
    if 5  <= h < 10: return "Morning_Rush"
    if 10 <= h < 16: return "Midday"
    if 16 <= h < 20: return "Evening_Rush"
    return "Night"

data["Time_Period"]   = data["Hour"].apply(time_of_day)
data["Total_Vehicles"] = data["Car_Count"] + data["Bike_Count"] + data["Truck_Count"]
data["Truck_Ratio"]    = data["Truck_Count"] / (data["Total_Vehicles"] + 1)
data["Rush_x_Density"] = (
    data["Time_Period"].isin(["Morning_Rush", "Evening_Rush"]).astype(int) *
    data["Traffic_Density"].map({"Low": 1, "Medium": 2, "High": 3})
)

data.drop(columns=["Timestamp"], inplace=True)
print("✅ Feature engineering done")

# ── 3. Define feature types ──────────────────────────────────────────────────
categorical_features = ["Day", "Weather", "Area_Type", "Traffic_Density", "Time_Period"]
numerical_features   = [
    "Hour", "Month", "Is_Weekend",
    "Car_Count", "Bike_Count", "Truck_Count",
    "Total_Vehicles", "Truck_Ratio", "Rush_x_Density"
]

# Ordinal encode all categoricals
ordinal_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
preprocessor = ColumnTransformer([
    ("cat", ordinal_enc, categorical_features),
    ("num", "passthrough", numerical_features)
])

# ── 4. Split ─────────────────────────────────────────────────────────────────
X = data[categorical_features + numerical_features]
y = data["Signal_Waiting_Time_Seconds"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 5. Train GradientBoosting ─────────────────────────────────────────────────
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
print("✅ Model trained")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n📊 Results:")
print(f"   RMSE : {rmse:.2f} seconds")
print(f"   MAE  : {mae:.2f} seconds")
print(f"   R²   : {r2:.4f}")

# ── 7. Save model + metadata ──────────────────────────────────────────────────
joblib.dump(pipeline, "traffic_model.pkl")

metadata = {
    "features": categorical_features + numerical_features,
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "metrics": {"rmse": round(rmse, 2), "mae": round(mae, 2), "r2": round(r2, 4)},
    "model_type": "GradientBoostingRegressor"
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Model saved as traffic_model.pkl")
print("✅ Metadata saved as model_metadata.json")
print("🎉 TRAINING COMPLETE")
