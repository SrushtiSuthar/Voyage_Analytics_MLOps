# src/training.py
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import pandas as pd
import numpy as np

from src.data_handler import load_csv, save_features
from src.data_wrangling import wrangle_flights
from src.data_preprocessing import preprocess_flights, data_spliting_Xy, data_spliting_X
from src.models import FlightPriceModel
from sklearn.model_selection import train_test_split


def train_flight_price_model(experiment_name: str = "voyage_flight_price") -> Dict:
    """
    Complete training pipeline for flight price model.
    Loads wrangled data → preprocess → train → evaluate → save.
    """
    # Paths
    PROJECT_ROOT = Path.cwd().parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw"

    # Load raw data
    print("Loading raw data...")
    flights = load_csv(DATA_DIR / "flights.csv")
    print(f"Flight rows loaded: {len(flights)}")

    # wrangling
    print("Wrangling...")
    flights_wrangled = wrangle_flights(flights)

    # Preprocess + feature engineering
    print("Preprocessing...")
    flights_prep = preprocess_flights(flights_wrangled)
    X, y = data_spliting_Xy(flights_prep, 
                         target_col = "price", 
                         drop_cols = ["travelcode", "usercode", "flighttype", "time", "distance", "date"]
                         )
    print(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

    # save features
    save_features(X, name="flights")
    print("Features saved")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Train all models
    results = {}
    for model_name in ["ridge", "random_forest", "gradient_boosting"]:
        print(f"\nTraining {model_name}...")
        result = FlightPriceModel.train_and_evaluate(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            mlflow_run_name=f"flight_{model_name}",
        )
        results[model_name] = result

    # 6) Best model summary
    best_model = min(results.keys(), key=lambda k: results[k]["test_rmse"])
    print(f"\nBest model: {best_model}")
    print(f"Test RMSE: {results[best_model]['test_rmse']:.2f}")
    print(f"Test R²: {results[best_model]['test_r2']:.3f}")

    return {
        "best_model": best_model,
        "results": results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

if __name__ == "__main__":
    # Run full pipeline
    results = train_flight_price_model()
