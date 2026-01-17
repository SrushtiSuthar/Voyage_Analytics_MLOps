from flask import Flask, request, jsonify
from pathlib import Path
import sys, os
import joblib
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import your functions (adjust names to match your files)
from src.data_preprocessing import preprocess_flights, data_spliting_X  # your functions
from src.data_wrangling import wrangle_flights

app = Flask(__name__)

# Load model
MODELS_DIR = PROJECT_ROOT / "models"
model_path = MODELS_DIR / "random_forest_flight.pkl"  # your best model
model = joblib.load(model_path)
print(f"✅ Model loaded: {model_path}")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    print("🔥 PREDICT REQUEST RECEIVED")
    try:
        input_data = request.get_json()
        print("Input data:", input_data)

        # Create single-row DataFrame
        df_input = pd.DataFrame([input_data])

        # Apply wrangling
        df_wrangle = wrangle_flights(df_input)

        # Apply preprocessing
        df_prep = preprocess_flights(df_wrangle)
        print("df_prep columns:", df_prep.columns.tolist())

        # Build features
        X = data_spliting_X(df_prep)
        print("X shape:", X.shape)
        print("X columns:", X.columns.tolist())

        # Predict
        pred = model.predict(X)[0]
        print("Prediction made:", pred)

        return jsonify({
            "status": "success",
            "predicted_price": float(pred),
            "price_range": [float(pred * 0.9), float(pred * 1.1)]
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
