import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import folium
from streamlit_folium import st_folium

#Page Setup 
st.set_page_config(page_title="NYC Taxi Trip Duration", layout="wide")

#Load Model 
@st.cache_resource
def load_model():
    return joblib.load("taxi_duration_model.pkl", mmap_mode='r')

model = load_model()

#Haversine 
def distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --- Session Init ---
if "pickup" not in st.session_state:
    st.session_state.pickup = None
if "dropoff" not in st.session_state:
    st.session_state.dropoff = None

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚖 NYC Taxi Trip Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Click on the map to set pickup and dropoff points.</p>", unsafe_allow_html=True)
st.divider()

# --- 2 Column Layout: Map + Controls ---
left, right = st.columns([2, 1])

# ----------------- LEFT SIDE (MAP) ----------------- #
with left:
    click_mode = st.radio("🖱️ Click Mode", ["Pickup", "Dropoff"], horizontal=True)

    # Map
    m = folium.Map(location=[40.75, -73.98], zoom_start=12)

    if st.session_state.pickup:
        folium.Marker(st.session_state.pickup, popup="Pickup", icon=folium.Icon(color="green")).add_to(m)
    if st.session_state.dropoff:
        folium.Marker(st.session_state.dropoff, popup="Dropoff", icon=folium.Icon(color="red")).add_to(m)

    if st.session_state.pickup and st.session_state.dropoff:
        folium.PolyLine([st.session_state.pickup, st.session_state.dropoff], color="orange", weight=3).add_to(m)

    map_data = st_folium(m, height=550, width=1200)

    if map_data and map_data["last_clicked"]:
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        if click_mode == "Pickup":
            st.session_state.pickup = [lat, lon]
            st.success(f"✅ Pickup: {lat:.5f}, {lon:.5f}")
        else:
            st.session_state.dropoff = [lat, lon]
            st.success(f"✅ Dropoff: {lat:.5f}, {lon:.5f}")

# ----------------- RIGHT SIDE (Inputs + Predict) ----------------- #
with right:
    st.subheader("📦 Ride Info")
    vendor_id = st.selectbox("Vendor ID", [1, 2])
    passenger_count = st.slider("Passenger Count", 1, 6, 1)
    date = st.date_input("Pickup Date", value=datetime.now().date())
    time = st.time_input("Pickup Time", value=datetime.now().time())
    pickup_datetime = datetime.combine(date, time)

    st.divider()
    st.subheader("🔮 Prediction")

    if st.button("🚀 Predict Duration"):
        if not st.session_state.pickup or not st.session_state.dropoff:
            st.warning("⚠️ Set both pickup and dropoff.")
        else:
            p_lat, p_lon = st.session_state.pickup
            d_lat, d_lon = st.session_state.dropoff
            hour = pickup_datetime.hour
            day = pickup_datetime.weekday()
            distance_km = distance(p_lat, p_lon, d_lat, d_lon)
            features = [[passenger_count, vendor_id, hour, day, distance_km]]

            try:
                prediction = model.predict(features)[0]
                minutes = int(prediction // 60)
                seconds = int(prediction % 60)
                st.markdown(
                    f"""
                    <div style='background-color:#00000;padding:10px;border-radius:8px;text-align:center;'>
                        <h3>🕒 {minutes} min {seconds} sec</h3>
                        <small>({int(prediction)} seconds total)</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# --- Footer ---
# --- Title with Inline Footer ---
st.markdown("""
    <div style='text-align: center;'>
        <p style='margin-top: 128px; font-size: 18px; color: #555;'>
            👨‍💻 Created by <strong style="color:#4CAF50;">Aly Osama and Aya Sayed</strong>
    </div>
""", unsafe_allow_html=True)

