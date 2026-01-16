from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_preprocessing import preprocess_flights, data_spliting_X  # your functions
from src.prediction import predict_flight_price  # if you have it

app = Flask(__name__)

# Load production model
MODELS_DIR = PROJECT_ROOT / "models"
model_path = MODELS_DIR / "random_forest_flight.pkl"  # or your best model
model = joblib.load(model_path)
print(f"Loaded model: {model_path}")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        
        # Validate input
        required = ["from", "to", "flighttype", "agency", "distance", "time", "date"]
        if not all(k in input_data for k in required):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Single prediction
        result = predict_flight_price(model_path, input_data)
        
        return jsonify({
            "status": "success",
            "predicted_price": result["predicted_price"],
            "confidence": getattr(result, "confidence", 0.95),
            "price_range": result.get("price_range", [0, 0])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
