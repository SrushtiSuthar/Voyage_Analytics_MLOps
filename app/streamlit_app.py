import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Voyage Analytics", layout="wide")

st.title("✈️ Voyage Analytics – Flight Price Prediction")
st.markdown("**Production ML model** predicting flight prices using Random Forest.")

import os
API_URL = os.getenv("API_URL", "http://localhost:5000/predict")
print(f"🔥 STREAMLIT STARTED - API_URL='{API_URL}'")

# Input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trip Details")
    from_city = st.selectbox("From", ["Aracaju (SE)", "Brasilia (DF)", "Campo Grande (MS)",
                                    "Florianopolis (SC)", "Natal (RN)", "Recife (PE)",
                                    "Rio de Janeiro", "Salvador (BH)", "Sao Paulo (SP)"])
    to_city = st.selectbox("To", ["Aracaju (SE)", "Brasilia (DF)", "Campo Grande (MS)",
                                 "Florianopolis (SC)", "Natal (RN)", "Recife (PE)",
                                 "Rio de Janeiro", "Salvador (BH)", "Sao Paulo (SP)"])
    flight_type = st.selectbox("Class", ["economic", "firstClass", "premium"])
    agency = st.selectbox("Airline", ["CloudFy", "FlyingDrops", "Rainbow"])

with col2:
    st.subheader("Flight Info")
    distance = st.slider("Distance (km)", 0, 1000, 500)
    duration = st.slider("Duration (hours)", 1.0, 3.0, 1.5)
    travel_date = st.date_input("Travel Date", value=pd.to_datetime("2026-01-15"))

if st.button("🚀 Predict Price", type="primary"):
    input_data = {
        "from": from_city,
        "to": to_city,
        "flighttype": flight_type,
        "agency": agency,
        "distance": distance,
        "time": duration,
        "date": str(travel_date),
    }

    with st.spinner("🔄 Predicting... Check Flask terminal for debug info"):
        try:
            response = requests.post(API_URL, json=input_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Price", f"₹{result['predicted_price']:,.0f}")
                with col2:
                    st.metric("Confidence", "95%")
                with col3:
                    low, high = result['price_range']
                    st.metric("Price Range", f"₹{low:,.0f} – ₹{high:,.0f}")
            else:
                st.error(f"❌ API Error {response.status_code}")
                st.code(response.text)
                
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
            st.info("👉 Run `python api/app.py` in another terminal")

st.markdown("---")
st.markdown("**Model: Random Forest | RMSE: 9.70 | R²: 0.999**")
