from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import logging

# =========================
# Logging configuration
# =========================
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# Initialize app
# =========================
app = FastAPI(title="Sonar Rock vs Mine Prediction API")

# Load model & scaler
model = joblib.load("sonar_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Input schema with validation
# =========================
class SonarInput(BaseModel):
    features: list = Field(..., description="List of 60 sonar features")

    @validator("features")
    def check_feature_length(cls, v):
        if len(v) != 60:
            raise ValueError("Exactly 60 features are required")
        return v


# =========================
# Root endpoint
# =========================
@app.get("/")
def home():
    return {"message": "Sonar Prediction API is running"}


# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
def predict(input_data: SonarInput):
    try:
        # Convert input to numpy array
        data = np.array(input_data.features).reshape(1, -1)

        # Scale
        data_scaled = scaler.transform(data)

        # Prediction
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled).max()

        result = "Mine" if prediction == "M" else "Rock"

        logging.info(
            f"Prediction made | Result: {result} | Confidence: {probability:.4f}"
        )

        return {
            "prediction": result,
            "confidence": round(float(probability), 4)
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
