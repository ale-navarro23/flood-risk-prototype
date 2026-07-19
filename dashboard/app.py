"""Streamlit dashboard for the flood risk prototype. Lets you pick a model and predict."""

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

TRAIN_RISK_THRESHOLD_M = 0.806
BAND_COLOR = {"Low": "green", "Moderate": "orange", "High": "red"}
BAND_HEX = {"Low": "#12a150", "Moderate": "#e0982b", "High": "#d64545"}

STATION = {"name": "Murray Bridge", "id": "A4261162", "lat": -35.12, "lon": 139.27}
DEFAULT_LEVELS = [0.67, 0.65, 0.65, 0.67, 0.69, 0.69, 0.73]
FEATURES = ["level_lag1", "level_lag2", "level_roll7", "level_change3"]


@st.cache_resource
def load_models():
    import joblib
    nb = Path(__file__).resolve().parent.parent / "notebooks"
    models = {}
    for name, fname in [("Random Forest", "Random_Forest.joblib"),
                        ("Logistic Regression", "logistic_regression_real.joblib")]:
        path = nb / fname
        if path.exists():
            m = joblib.load(path)
            models[name] = (m, list(getattr(m, "feature_names_in_", FEATURES)))
    return models


def features_from_levels(levels):
    window = levels[-7:]
    return {
        "level_lag1": levels[-1],
        "level_lag2": levels[-2],
        "level_roll7": sum(window) / len(window),
        "level_change3": levels[-1] - levels[-4],
    }


def band_of(p):
    return "High" if p >= 0.66 else "Moderate" if p >= 0.33 else "Low"


st.set_page_config(page_title="River Murray Flood Risk", page_icon="🌊", layout="wide")
st.title("🌊 River Murray Flood Risk Prototype")
st.caption("Next-day flood risk (high river level) for Murray Bridge. Advisory only.")

models = load_models()
if not models:
    st.error("No model found. Run the Random Forest notebook to create notebooks/Random_Forest.joblib.")
    st.stop()

with st.sidebar:
    st.header("Model")
    choice = st.selectbox("Choose a model", list(models.keys()))
    model, feature_order = models[choice]

    st.header("Recent river levels (m)")
    levels = []
    for i, d in enumerate(DEFAULT_LEVELS):
        levels.append(st.slider(f"Day -{len(DEFAULT_LEVELS)-i}", 0.0, 3.0, float(d), 0.01))
    shift = st.slider("Shift whole series (m)", -0.5, 2.0, 0.0, 0.05)
    levels = [round(x + shift, 3) for x in levels]

feats = features_from_levels(levels)
row = pd.DataFrame([[feats[c] for c in feature_order]], columns=feature_order)
prob = float(model.predict_proba(row)[0][1])
band = band_of(prob)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Prediction")
    st.metric("Flood probability", f"{prob*100:.0f}%")
    st.markdown(
        f"<div style='padding:10px 14px;border-radius:8px;color:white;"
        f"background:{BAND_HEX[band]};font-weight:600;display:inline-block'>Risk band: {band}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Model: {choice}")
    if band == "High":
        st.error("High risk. Would require operator review before any alert is sent.")
    st.markdown("**Model features (derived)**")
    st.table({f: [round(feats[f], 3)] for f in feature_order})
    st.caption(f"Training risk threshold: {TRAIN_RISK_THRESHOLD_M} m (0.80 quantile).")

with col2:
    st.subheader("Station")
    fmap = folium.Map(location=[STATION["lat"], STATION["lon"]], zoom_start=9, tiles="CartoDB positron")
    folium.CircleMarker(
        location=[STATION["lat"], STATION["lon"]],
        radius=12, color=BAND_COLOR[band], fill=True, fill_color=BAND_COLOR[band], fill_opacity=0.9,
        popup=f"{STATION['name']}: {band} ({prob*100:.0f}%)", tooltip=STATION["name"],
    ).add_to(fmap)
    st_folium(fmap, height=430, use_container_width=True)

st.divider()
st.caption("Prototype for academic assessment (ITA602). Not for operational flood-warning use.")
