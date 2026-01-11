from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.data_wrangling import wrangle_flights
from src.data_preprocessing import preprocess_flights, data_spliting

def predict_flight_price(
    model_path: str | Path,
    input_data: Dict[str, Any],
    encoders_path: str | Path = None,
    scaler_path: str | Path = None,
) -> Dict[str, float]:
    """
    Make a flight price prediction given input features.

    input_data: dict with keys like 'from', 'to', 'distance', 'time', etc.
    Returns prediction + confidence interval.
    """

    model = joblib.load(model_path)

    # Create minimal DataFrame
    df_input = pd.DataFrame([input_data])

    # Apply same wrangling
    df_wrangle = wrangle_flights(df_input)

    # Apply same preprocessing as training
    df_prep = preprocess_flights(df_wrangle)

    # Build features (same logic as training)
    X, _ = data_spliting(df_prep)

    # Predict
    prediction = model.predict(X)[0]

    # Confidence (for tree models: use std of leaf predictions)
    if hasattr(model, "predict_proba"):
        # Regression: std of predictions across trees
        predictions = np.array([est.predict(X) for est in model.estimators_]).flatten()
        confidence = np.std(predictions)
    else:
        confidence = 0.95  # fallback

    return {
        "predicted_price": float(prediction),
        "confidence": float(confidence),
        "price_range": [prediction * 0.9, prediction * 1.1],
    }

def batch_predict_flight_price(model_path: str | Path, X_batch: pd.DataFrame) -> np.ndarray:
    # Batch predictions
    model = joblib.load(model_path)
    return model.predict(X_batch)
