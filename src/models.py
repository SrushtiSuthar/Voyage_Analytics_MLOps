# imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd

class FlightPriceModel:
    # Flight price regression model manager

    MODELS = {
        "ridge"             : Ridge(alpha=1.0),
        "random_forest"     : RandomForestRegressor(n_estimators=100, 
                                                    max_depth=15, 
                                                    random_state=42, 
                                                    n_jobs=-1
                                                    ),
        "gradient_boosting" : GradientBoostingRegressor(n_estimators=100, 
                                                        learning_rate=0.1, 
                                                        max_depth=6, 
                                                        random_state=42),
                                                        }

    @staticmethod
    def train_and_evaluate(model_name: str, X_train, y_train, X_test, y_test, mlflow_run_name: str = None):
        # Train a model and log to MLflow.

        model = FlightPriceModel.MODELS[model_name].__class__(**FlightPriceModel.MODELS[model_name].get_params())

        with mlflow.start_run(run_name=mlflow_run_name or model_name):
            # Train
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Log to MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)

            # Save model
            mlflow.sklearn.log_model(model, "model")
            joblib.dump(model, f"models/{model_name}_flight.pkl")

            return {
                "model": model,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_mae": train_mae,
                "test_mae": test_mae,
            }

    @staticmethod
    def load_model(model_name: str):
        # Load a trained model
        return joblib.load(f"models/{model_name}_flight.pkl")

    @staticmethod
    def predict(model, X: pd.DataFrame) -> np.ndarray:
        # Make predictions
        return model.predict(X)
