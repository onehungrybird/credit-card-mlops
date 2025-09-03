from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow.pyfunc
import os
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
scaler = None

# Feature columns (must match training)
FEATURE_COLUMNS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
    'V28', 'Amount'
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler on startup."""
    global model, scaler
    try:
        model_path = "models:/credit-card-fraud-model/Staging"
        scaler_path = "models/scaler.joblib"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        model = mlflow.pyfunc.load_model(model_path)
        scaler = joblib.load(scaler_path)

        logger.info("Model and scaler loaded successfully.")
        logger.info("API is ready to serve predictions.")

    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise e

    yield

    logger.info("Shutting down: clearing resources.")
    model = None
    scaler = None

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0",
    lifespan=lifespan
)

# Request model
class TransactionInput(BaseModel):
    Time: Optional[float] = 0.0
    Amount: Optional[float] = 0.0
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

# Response model
class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    confidence: str

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/")
async def root():
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Service not ready: model not loaded")

    try:
        input_data = pd.DataFrame([transaction.dict()])

        # Keep all features your model was trained on
        model_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        input_for_model = input_data[model_features]

        # Use model.predict() for PyFuncModel
        pred = model.predict(input_for_model)

        # If the model returns probabilities in a 2D array
        if hasattr(pred[0], "__len__"):
            fraud_probability = float(pred[0][1])
        else:
            fraud_probability = float(pred[0])

        is_fraud = fraud_probability > 0.5

        confidence = (
            "high" if fraud_probability < 0.3 or fraud_probability > 0.7
            else "medium" if fraud_probability < 0.4 or fraud_probability > 0.6
            else "low"
        )

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_probability, 4),
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


def start_server():
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
