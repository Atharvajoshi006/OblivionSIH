import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

os.makedirs("models", exist_ok=True)
np.random.seed(42)
n_records = 600

rainfall = np.random.exponential(scale=12.0, size=n_records)
soil_moisture = np.clip(25.0 + rainfall * 1.8 + np.random.normal(0, 4, n_records), 15.0, 95.0)
active_shovels = np.random.randint(3, 10, size=n_records)
haul_trucks = active_shovels * 2 + np.random.randint(0, 4, size=n_records)
shift_blasting = np.random.choice([0, 1], size=n_records, p=[0.25, 0.75])

planned_tonnage = active_shovels * 300 + haul_trucks * 75 + shift_blasting * 400
weather_penalty = (rainfall * 18.5) + (soil_moisture * 4.2)
equipment_penalty = np.where(haul_trucks < (active_shovels * 2), 250, 0)

actual_tonnage = planned_tonnage - weather_penalty - equipment_penalty + np.random.normal(0, 30, n_records)
shortfall_tons = np.maximum(0, planned_tonnage - actual_tonnage)

df = pd.DataFrame({
    'rainfall_mm': rainfall,
    'soil_moisture_pct': soil_moisture,
    'active_shovels': active_shovels,
    'haul_trucks': haul_trucks,
    'shift_blasting': shift_blasting,
    'shortfall_tons': shortfall_tons
})
df.to_csv("data/mine_logs.csv", index=False)

features = ['rainfall_mm', 'soil_moisture_pct', 'active_shovels', 'haul_trucks', 'shift_blasting']
X = df[features]
y = df['shortfall_tons']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=75, max_depth=3, learning_rate=0.08, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Model trained. Mean Absolute Error: {mae:.2f} Tons")

joblib.dump(model, "models/shortfall_xgb.pkl")