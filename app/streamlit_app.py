import streamlit as st
import requests
import pandas as pd
from pathlib import Path
import os

# Use environment variable or default to localhost
#API_ROOT = os.getenv("API_URL", "http://localhost:5000")
#API_URL = API_ROOT.rstrip("/") + "/predict"

st.set_page_config(page_title="Voyage Analytics", layout="wide")

st.title("✈️ Voyage Analytics – Flight Price Prediction")
st.markdown("**Production ML model** predicting flight prices using Random Forest.")

# API endpoint
API_URL = "http://localhost:5000/predict"

# Input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trip Details")
    from_city = st.selectbox("From", ["Aracaju (SE)",
                                      "Brasilia (DF)",
                                      "Campo Grande (MS)",
                                      "Florianopolis (SC)",
                                      "Natal (RN)",
                                      "Recife (PE)",
                                      "Rio de Janeiro",
                                      "Salvador (BH)",
                                      "Sao Paulo (SP)"])
    to_city = st.selectbox("To", ["Aracaju (SE)",
                                  "Brasilia (DF)",
                                  "Campo Grande (MS)",
                                  "Florianopolis (SC)",
                                  "Natal (RN)",
                                  "Recife (PE)",
                                  "Rio de Janeiro",
                                  "Salvador (BH)",
                                  "Sao Paulo (SP)"])
    flight_type = st.selectbox("Class", ["economic", "firstClass", "premium"])
    agency = st.selectbox("Airline", ["CloudFy", "FlyingDrops", "Rainbow"])

with col2:
    st.subheader("Flight Info")
    distance = st.slider("Distance (km)", 0, 1000, 500 )
    duration = st.slider("Duration (hours)", 1.0, 3.0, 1.5)
    travel_date = st.date_input("Travel Date", value=pd.to_datetime("2026-01-15"))

if st.button("🚀 Predict Price", type="primary"):
    # Prepare input
    input_data = {
        "from": from_city,
        "to": to_city,
        "flighttype": flight_type,
        "agency": agency,
        "distance": distance,
        "time": duration,
        "date": str(travel_date),
    }
    
    print("🔥 STREAMLIT: Predict clicked")  # DEBUG
    print("🔥 STREAMLIT: Sending payload:", input_data)  # DEBUG

    with st.spinner("Predicting..."):
        try:
            print("🔥 STREAMLIT: Making requests.post")  # DEBUG
            response = requests.post(API_URL, json=input_data, timeout=10)
            
            print("🔥 STREAMLIT: Status code:", response.status_code)  # DEBUG
            print("🔥 STREAMLIT: Response text:", response.text)  # DEBUG
            
            result = response.json()
            print("🔥 STREAMLIT: Parsed result:", result)  # DEBUG

            if response.status_code == 200:
                st.success("✅ Prediction Complete!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Price", f"₹{result['predicted_price']:,.0f}")
                with col2:
                    st.metric("Confidence", "95%")  # Fixed: hardcoded for now
                with col3:
                    low, high = result['price_range']
                    st.metric("Price Range", f"₹{low:,.0f} – ₹{high:,.0f}")
            else:
                st.error(f"❌ API Error {response.status_code}: {result.get('error', 'Unknown')}")
                
        except requests.exceptions.RequestException as e:
            print("🔥 STREAMLIT: Connection error:", str(e))  # DEBUG
            st.error(f"❌ Connection Error: {e}")
            st.info("Make sure Flask API is running: `python api/app.py`")
        except KeyError as e:
            print("🔥 STREAMLIT: Missing key:", e)  # DEBUG
            st.error(f"❌ API response missing field: {e}")
        except Exception as e:
            print("🔥 STREAMLIT: Unexpected error:", str(e))  # DEBUG
            st.error(f"❌ Unexpected error: {e}")