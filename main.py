from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Cyncly AI Innovation: Energy Predictor")

# Load your Random Forest model
model = joblib.load("final_model.pkl")

# Define the data structure for your 21 input columns
class EnergyInput(BaseModel):
    building_id: int
    meter: int
    site_id: int
    primary_use: str  # e.g., "Education", "Office"
    square_feet: float
    year_built: float
    air_temperature: float
    cloud_coverage: float
    dew_temperature: float
    precip_depth_1_hr: float
    sea_level_pressure: float
    wind_direction: float
    wind_speed: float
    hour: int
    day: int
    weekday: int
    month: int
    lag_1h: float
    lag_24h: float
    lag_48h: float

@app.post("/predict")
def predict_energy(data: EnergyInput):
    # Convert input to a DataFrame
    # Note: If your model was trained on numerical values for 'primary_use', 
    # you may need to add a LabelEncoder step here.
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Make predictions for 1h, 24h, and 48h
    prediction = model.predict(input_df)
    
    return {
        "status": "success",
        "predictions": {
            "energy_1h": float(prediction[0][0]),
            "energy_24h": float(prediction[0][1]),
            "energy_48h": float(prediction[0][2])
        }
    }