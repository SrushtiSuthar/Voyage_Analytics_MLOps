from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.data_wrangling import wrangle_flights
from src.data_preprocessing import preprocess_flights, data_spliting_X

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

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
    print("Model loaded")

    # Create minimal DataFrame
    df_input = pd.DataFrame([input_data])
    print("Sample input loaded")

    # Apply same wrangling
    df_wrangle = wrangle_flights(df_input)
    print("Sample input wrangled")

    # Apply same preprocessing as training
    df_prep = preprocess_flights(df_wrangle)
    print("Sample input preprocessed")
    print(df_prep.columns.tolist())

    # Build features (same logic as training)
    X = data_spliting_X(df_prep)

    # Load training feature columns
    feature_cols_path = FEATURE_DIR / "flights_features.pkl"
    X_train = joblib.load(feature_cols_path)
    print("TRAIN columns:", len(X_train.columns))
    print(X_train.columns.tolist()[:20])
    feature_cols = list(X_train.columns)

    # Align columns: add missing, drop extra
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X.drop(
        columns=["travelcode", "usercode", "flighttype", "time", "distance", "date"],
        errors="ignore"
        )

    # Reorder columns exactly as training
    X = X[feature_cols]
    print(X)

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
