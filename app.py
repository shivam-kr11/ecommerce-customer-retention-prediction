
#=======================
#       Import Libraries
#=======================

from pathlib import Path
import pandas as pd
import joblib
from fastapi import FastAPI


#=======================
#       Initialize API
#=======================

app = FastAPI(
    title="E-commerce Customer Retention API",
    description="API for customer classification using a trained Decision Tree.",
    version="1.0.0"
)


#=======================
#       Load Model Artifacts
#=======================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "retention_model.pkl"

artifacts = joblib.load(MODEL_PATH)

model = artifacts["model"]
encoder = artifacts["encoder"]
scaler = artifacts["scaler"]
numeric_columns = artifacts["numeric_columns"]
categorical_columns = artifacts["categorical_columns"]
feature_columns = artifacts["feature_columns"]
threshold = artifacts["threshold"]


#=======================
#       Home Endpoint
#=======================

@app.get("/")
def home():
    return {
        "message": "E-commerce Customer Retention API is running",
        "docs": "/docs"
    }


#=======================
#       Model Information
#=======================

@app.get("/model-info")
def model_info():
    return {
        "model": type(model).__name__,
        "threshold": float(threshold),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "total_feature_count": len(feature_columns)
    }


#=======================
#       Prediction Input Schema
#=======================

from pydantic import BaseModel
from typing import Any


class PredictionRequest(BaseModel):
    features: dict[str, Any]


#=======================
#       Prediction Endpoint
#=======================

@app.post("/predict")
def predict_customer(request: PredictionRequest):

    raw_features = pd.DataFrame([request.features])

    # Check required input features
    required_columns = numeric_columns + categorical_columns
    missing_columns = [
        col for col in required_columns
        if col not in raw_features.columns
    ]

    if missing_columns:
        return {
            "error": "Missing required features",
            "missing_features": missing_columns
        }

    # Arrange columns in training order
    raw_features = raw_features[required_columns]

    # Encode categorical features
    encoded_array = encoder.transform(
        raw_features[categorical_columns]
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_columns)
    )

    # Combine numeric and encoded features
    numeric_df = raw_features[numeric_columns].reset_index(drop=True)
    combined = pd.concat([numeric_df, encoded_df], axis=1)
    combined = combined[feature_columns]

    # Scale features
    scaled_features = scaler.transform(combined)

    # Predict
    probability = float(model.predict_proba(scaled_features)[0, 1])
    prediction = int(probability >= threshold)

    
       
    return {
        "prediction": prediction,
        "predicted_label": (
            "Class 1" if prediction == 1 else "Class 0"
        ),
        "probability_class_1": round(probability, 4),
        "threshold": float(threshold)
    }

    
