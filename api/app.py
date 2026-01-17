from flask import Flask, request, jsonify
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
import traceback

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Your imports (adjust if filenames differ)
try:
    from src.data_preprocessing import preprocess_flights, data_spliting_X
    from src.data_wrangling import wrangle_flights
    print("✅ All preprocessing modules imported")
except ImportError as e:
    print("❌ Import error:", e)
    # Dummy functions if imports fail
    def preprocess_flights(df): return df
    def data_spliting_X(df): return df
    def wrangle_flights(df): return df

app = Flask(__name__)

# Load model
MODELS_DIR = PROJECT_ROOT / "models"
model_path = MODELS_DIR / "random_forest_flight.pkl"
model = joblib.load(model_path)
print(f"✅ Model loaded: {model_path}")
print(f"✅ Expected features: {model.feature_names_in_.tolist()}")

@app.route("/health", methods=["GET"])
def health():
    print("🔥 HEALTH CALLED")
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    print("🔥 PREDICT CALLED")
    try:
        data = request.get_json()
        print("Input:", data)
        
        df_input = pd.DataFrame([data])
        
        # Processing pipeline
        df_wrangle = wrangle_flights(df_input)
        df_prep = preprocess_flights(df_wrangle)
        X = data_spliting_X(df_prep)
        
        # FIX: Align with training features
        expected_features = model.feature_names_in_
        X = X.reindex(columns=expected_features, fill_value=0)
        
        pred = model.predict(X)[0]
        print(f"✅ Prediction: {pred}")
        
        return jsonify({
            "status": "success",
            "predicted_price": float(pred),
            "price_range": [float(pred * 0.9), float(pred * 1.1)]
        })
        
    except Exception as e:
        print("❌ ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
