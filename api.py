
from pathlib import Path
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("AgriVisionAI")


# Load the trained model
MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="AgriVision AI API",
    description="Crop production prediction API",
    version="1.0.0"
)


# Input validation
class PredictionInput(BaseModel):
    State_Name: str = Field(min_length=2)
    District_Name: str = Field(min_length=2)
    Crop_Year: int = Field(ge=1900, le=2100)
    Season: str = Field(min_length=2)
    Crop: str = Field(min_length=2)
    Area: float = Field(gt=0)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>AgriVision AI API</title>
        </head>
        <body style="font-family: Arial; padding: 40px;">
            <h1>AgriVision AI</h1>
            <h2>Crop Production Prediction API</h2>
            <p>API is running successfully.</p>
            <p>Visit <a href="/docs">/docs</a> to test predictions.</p>
        </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "DecisionTreeRegressor"
    }


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Log prediction input
        logger.info(
            "Prediction request received | State: %s | District: %s | "
            "Year: %s | Season: %s | Crop: %s | Area: %s",
            data.State_Name,
            data.District_Name,
            data.Crop_Year,
            data.Season,
            data.Crop,
            data.Area
        )

        input_data = pd.DataFrame([{
            "State_Name": data.State_Name,
            "District_Name": data.District_Name,
            "Crop_Year": data.Crop_Year,
            "Season": data.Season,
            "Crop": data.Crop,
            "Area": data.Area
        }])

        prediction = model.predict(input_data)[0]

        predicted_value = round(float(prediction), 2)

        # Log prediction result
        logger.info(
            "Prediction successful | Crop: %s | Predicted Production: %s",
            data.Crop,
            predicted_value
        )

        return {
            "status": "success",
            "predicted_production": predicted_value,
            "unit": "dataset production units"
        }

    except Exception as error:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
