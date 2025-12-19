import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

import folium
from streamlit_folium import st_folium

import osmnx as ox
import networkx as nx
from geopy.distance import geodesic

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="NYC Taxi Trip Duration", layout="wide")

# --------------------------------------------------
# Load ML Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("taxi_duration_model.pkl", mmap_mode="r")

model = load_model()

# --------------------------------------------------
# Load NYC Road Graph
# --------------------------------------------------
@st.cache_resource
def load_graph():
    return ox.graph_from_place(
        "Manhattan, New York City, USA",
        network_type="drive"
    )
G = load_graph()  

# --------------------------------------------------
# A* Routing Function
# --------------------------------------------------
def astar_route(G, pickup, dropoff):
    orig = ox.nearest_nodes(G, pickup[1], pickup[0])
    dest = ox.nearest_nodes(G, dropoff[1], dropoff[0])

    route = nx.astar_path(
        G,
        orig,
        dest,
        heuristic=lambda u, v: geodesic(
            (G.nodes[u]["y"], G.nodes[u]["x"]),
            (G.nodes[v]["y"], G.nodes[v]["x"])
        ).meters,
        weight="length"
    )

    route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

    route_km = sum(
        geodesic(route_coords[i], route_coords[i + 1]).km
        for i in range(len(route_coords) - 1)
    )

    return route_coords, route_km

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "pickup" not in st.session_state:
    st.session_state.pickup = None

if "dropoff" not in st.session_state:
    st.session_state.dropoff = None

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#4CAF50;'>🚖 NYC Taxi Trip Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Click on the map to select pickup and dropoff points</p>",
    unsafe_allow_html=True
)
st.divider()

# --------------------------------------------------
# Layout
# --------------------------------------------------
left, right = st.columns([2, 1])

# ================= LEFT: MAP ======================
with left:
    click_mode = st.radio("🖱️ Click Mode", ["Pickup", "Dropoff"], horizontal=True)

    m = folium.Map(location=[40.75, -73.98], zoom_start=12)

    if st.session_state.pickup:
        folium.Marker(
            st.session_state.pickup,
            popup="Pickup",
            icon=folium.Icon(color="green")
        ).add_to(m)

    if st.session_state.dropoff:
        folium.Marker(
            st.session_state.dropoff,
            popup="Dropoff",
            icon=folium.Icon(color="red")
        ).add_to(m)

    # Draw A* route
    if st.session_state.pickup and st.session_state.dropoff:
        try:
            route_coords, _ = astar_route(
                G, st.session_state.pickup, st.session_state.dropoff
            )
            folium.PolyLine(
                route_coords,
                color="blue",
                weight=4,
                tooltip="A* Route"
            ).add_to(m)
        except:
            pass

    map_data = st_folium(m, height=550, width=1200)

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]

        if click_mode == "Pickup":
            st.session_state.pickup = [lat, lon]
            st.success(f"✅ Pickup set: {lat:.5f}, {lon:.5f}")
        else:
            st.session_state.dropoff = [lat, lon]
            st.success(f"✅ Dropoff set: {lat:.5f}, {lon:.5f}")

# ================= RIGHT: INPUTS ==================
with right:
    st.subheader("📦 Ride Information")

    vendor_id = st.selectbox("Vendor ID", [1, 2])
    passenger_count = st.slider("Passenger Count", 1, 6, 1)

    date = st.date_input("Pickup Date", datetime.now().date())
    time = st.time_input("Pickup Time", datetime.now().time())
    pickup_datetime = datetime.combine(date, time)

    hour = pickup_datetime.hour
    day = pickup_datetime.weekday()

    st.divider()
    st.subheader("🔮 Prediction")

    if st.button("🚀 Predict Duration"):
        if not st.session_state.pickup or not st.session_state.dropoff:
            st.warning("⚠️ Please select both pickup and dropoff points.")
        else:
            try:
                route_coords, distance_km = astar_route(
                    G, st.session_state.pickup, st.session_state.dropoff
                )

                features = [[
                    passenger_count,
                    vendor_id,
                    hour,
                    day,
                    distance_km
                ]]

                prediction = model.predict(features)[0]

                minutes = int(prediction // 60)
                seconds = int(prediction % 60)

                st.markdown(
                    f"""
                    <div style='background:#f0f0f0;padding:15px;
                                border-radius:10px;text-align:center;'>
                        <h2>🕒 {minutes} min {seconds} sec</h2>
                        <p>Distance: {distance_km:.2f} km</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div style='text-align:center;margin-top:120px;color:#555;'>
        👨‍💻 Created by <strong style='color:#4CAF50;'>Aly Osama & Aya Sayed</strong>
    </div>
    """,
    unsafe_allow_html=True
)


