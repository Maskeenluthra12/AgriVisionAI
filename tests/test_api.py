
import math
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


# Test 1: API health check
def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


# Test 2: API home page
def test_home_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "AgriVision AI" in response.text


# Test 3: Crop production prediction
def test_crop_prediction():
    payload = {
        "State_Name": "Punjab",
        "District_Name": "LUDHIANA",
        "Crop_Year": 2020,
        "Season": "Rabi",
        "Crop": "Wheat",
        "Area": 100
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "success"
    assert "predicted_production" in result
    assert math.isfinite(result["predicted_production"])


# Test 4: Invalid input validation
def test_invalid_prediction_input():
    payload = {
        "State_Name": "Punjab",
        "District_Name": "LUDHIANA",
        "Crop_Year": 2020,
        "Season": "Rabi",
        "Crop": "Wheat",
        "Area": -10
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
