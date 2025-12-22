# Voyage Analytics - MLOps in Travel

Production ML system predicting flight/hotel prices, user segmentation, and hotel recommendations.

## Quick Start

conda activate voyage_ml
pip install -r requirements.txt
dvc pull
mlflow ui --port 5000
jupyter lab --notebook-dir=notebooks

## Architecture

Data (DVC) → MLflow → FastAPI → Docker → Kubernetes → Streamlit

## Models
1. Flight Price Prediction (XGBoost, R²=0.85)
2. Hotel Price Prediction (Random Forest, RMSE=₹450)
3. User Gender Classification (Accuracy=87%)
4. Hotel Recommendation System (Content-based) 
