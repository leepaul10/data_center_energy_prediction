import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Initialize FastAPI
app = FastAPI(title="Data Center Energy Prediction API")

# 2. Add CORS Middleware (Essential for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your React app to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the Model
try:
    model = joblib.load("final_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 4. Define Input Schema (Matches your Postman Body)
class EnergyRequest(BaseModel):
    building_id: int
    meter: int
    meter_reading: float
    site_id: int
    primary_use: int
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

@app.get("/")
def health_check():
    return {"status": "Online", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: EnergyRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    try:
        # Convert incoming JSON to DataFrame
        df = pd.DataFrame([data.dict()])
        df = df.apply(pd.to_numeric)
        
        # Prediction
        prediction = model.predict(df)
        
        return {
            "forecast": {
                "t_plus_1h": round(float(prediction[0][0]), 2),
                "t_plus_24h": round(float(prediction[0][1]), 2),
                "t_plus_48h": round(float(prediction[0][2]), 2)
            },
            "unit": "kWh"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    