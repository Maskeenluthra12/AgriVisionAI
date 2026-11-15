# AgriVisionAI

**AI-Powered Crop Production Prediction System**

AgriVisionAI is an intelligent agriculture platform that uses Machine Learning to predict crop production based on agricultural parameters such as state, district, crop, season, year, and cultivated area.

The project combines machine learning, data analytics, a Streamlit frontend, and a FastAPI backend to provide an integrated crop production prediction system.
Feature	Type	Allowed range / values	Why it matters
State_Name	String	Valid state names	Geographical location
District_Name	String	Valid district names	Regional production
Crop_Year	Integer	1900–2100	Historical year
Season	String	Valid agricultural seasons	Seasonal variation
Crop	String	Valid crop names	Crop selection
Area	Float	Greater than 0	Cultivated area
Production	Float	Greater than or equal to 0	Target variable

---

## Key Features

- **Crop Production Prediction:** Predict crop production using a trained machine learning model.
- **Machine Learning:** Train and compare Decision Tree, Random Forest, and Extra Trees regression models.
- **Interactive Dashboard:** Explore agricultural information through a Streamlit interface.
- **Advanced Analytics:** Analyze crop production patterns and agricultural data.
- **AI Advisor:** Access AI-powered agricultural assistance.
- **Weather Integration:** Weather-related information through the weather service.
- **REST API:** Generate predictions through FastAPI endpoints.
- **Input Validation:** Validate prediction requests using Pydantic.
- **Prediction Logging:** Record prediction requests, results, and errors.
- **Docker Support:** Run the application using Docker and Docker Compose.
- **Automated Testing:** Test API functionality using Pytest.
- **CI Pipeline:** Run automated checks through GitHub Actions.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Experiment Tracking | MLflow |
| API Testing | Pytest |
| Data Validation | Pydantic |
| Containerization | Docker |
| Container Orchestration | Docker Compose |
| CI/CD | GitHub Actions |

---

## Machine Learning Approach

AgriVisionAI uses regression models to estimate crop production.

**Input Features:**
- State Name
- District Name
- Crop Year
- Season
- Crop
- Cultivated Area

**Target Variable:** Production

### Models Compared

1. Decision Tree Regressor
2. Random Forest Regressor
3. Extra Trees Regressor

### Model Evaluation

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The current training results are:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Decision Tree | 108563.729 | 4619473.135 | 0.9470 |
| Random Forest | 116964.282 | 4799229.714 | 0.9428 |
| Extra Trees | 122191.556 | 4950647.127 | 0.9391 |

**Selected Model:** Decision Tree Regressor

The selected model is saved at `models/model.joblib`.

*Note: These scores represent the current held-out test split and do not guarantee the same performance on unseen real-world data.*

---

## Project Structure

```text
AgriVisionAI/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .streamlit/
│   └── secrets.toml
│
├── assets/
│   ├── css/
│   │   └── style.css
│   └── images/
│       └── hero.jpg
│
├── datasets/
│   └── crop_production.csv
│
├── models/
│   ├── model.joblib
│   ├── model_comparison.csv
│   └── training_metrics.json
│
├── pages/
│   ├── Crop_Prediction.py
│   ├── Dashboard.py
│   ├── Advanced_Analytics.py
│   ├── AI_Advisor.py
│   ├── Reports.py
│   └── 4_Weather.py
│
├── services/
│   ├── gemini_service.py
│   └── weather_service.py
│
├── tests/
│   └── test_api.py
│
├── api.py
├── app.py
├── train.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-serve.txt
└── README.md
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Maskeenluthra12/AgriVisionAI.git
cd AgriVisionAI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The frontend will be available at:

`http://localhost:8501`

---

## Machine Learning Model Training

To train and compare the machine learning models:

```bash
python train.py
```

The training process saves the selected model and evaluation metrics inside the `models/` directory.

MLflow is used to track model training experiments.

---

## FastAPI Backend

Start the API server:

```bash
uvicorn api:app --reload
```

The backend will be available at:

`http://localhost:8000`

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict` | Crop production prediction |
| GET | `/docs` | Interactive API documentation |

### Sample Prediction Request

```json
{
  "State_Name": "Punjab",
  "District_Name": "LUDHIANA",
  "Crop_Year": 2020,
  "Season": "Rabi",
  "Crop": "Wheat",
  "Area": 100
}
```

---

## Docker Deployment

AgriVisionAI supports containerized deployment using Docker Compose.

Build and start the services:

```bash
docker compose up -d --build
```

Check running containers:

```bash
docker compose ps
```

Access the applications:

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

Stop the containers:

```bash
docker compose down
```

---

## Automated Testing

Run API tests using:

```bash
python -m pytest tests/test_api.py -v
```

The test suite covers:

- API health check
- API home page
- Successful crop prediction
- Invalid prediction input

---

## Continuous Integration

AgriVisionAI uses **GitHub Actions** to automate the development testing workflow.

The CI pipeline is configured to run on pushes and pull requests targeting the `main` branch.

The workflow includes:

1. Repository checkout
2. Python environment setup
3. Dependency installation
4. Machine learning model training
5. Automated API testing

---

## Future Enhancements

- Integration of additional agricultural datasets.
- Improved prediction performance through model experimentation.
- More detailed regional crop analytics.
- Enhanced weather-based agricultural recommendations.
- Further deployment and scalability improvements.

---

## Repository

**GitHub:** https://github.com/Maskeenluthra12/AgriVisionAI

---

*AgriVisionAI — Bringing Artificial Intelligence and Data Analytics into Agriculture.*
