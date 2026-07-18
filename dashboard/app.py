"""
Flood Risk Dashboard (Streamlit)
================================
River Murray flood-risk prototype — serves the team's trained Logistic
Regression model (next-day high-river-level risk for Murray Bridge).

Features:
- Risk gauge + colour-coded band
- Sensitivity band (how the score moves under +/- 5 cm gauge error)
- Explainability panel (exact per-feature contributions for logistic regression)
- 30-day river-level trend vs the risk threshold
- Scenario presets instead of raw sliders

Prediction source: the FastAPI backend if an API URL is set, otherwise the
local model bundled in the repo, otherwise a transparent fallback.
"""

import math
import os
from datetime import datetime
from pathlib import Path

import altair as alt
import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

TRAIN_RISK_THRESHOLD_M = 0.806
FEATURES = ["level_lag1", "level_lag2", "level_roll7", "level_change3"]
FEATURE_LABEL = {
    "level_lag1": "Level yesterday",
    "level_lag2": "Level 2 days ago",
    "level_roll7": "7-day average level",
    "level_change3": "3-day level change",
}
STATION = {"name": "Murray Bridge", "id": "A4261162", "lat": -35.12, "lon": 139.27}
BAND_HEX = {"Low": "#12a150", "Moderate": "#e0982b", "High": "#d64545"}
BAND_COLOR = {"Low": "green", "Moderate": "orange", "High": "red"}

# Real-data fallbacks (used if the CSV or model is unavailable at runtime).
FALLBACK_LEVELS = [0.725, 0.689, 0.731, 0.661, 0.653, 0.647, 0.616, 0.604, 0.654,
                   0.631, 0.592, 0.668, 0.644, 0.638, 0.667, 0.664, 0.686, 0.755,
                   0.733, 0.726, 0.713, 0.717, 0.718, 0.718, 0.7, 0.668, 0.648,
                   0.686, 0.694, 0.73]
FALLBACK_BASELINE = {"level_lag1": 0.693, "level_lag2": 0.693, "level_roll7": 0.698, "level_change3": -0.001}
FALLBACK_COEF = {"level_lag1": 13.095, "level_lag2": 3.991, "level_roll7": 9.148, "level_change3": 1.235}
FALLBACK_INTERCEPT = -20.165


@st.cache_resource
def load_model():
    here = Path(__file__).resolve().parent
    for p in [here.parent / "backend" / "models" / "logistic_regression_real.joblib",
              here.parent / "notebooks" / "logistic_regression_real.joblib"]:
        if p.exists():
            try:
                import joblib
                return joblib.load(p)
            except Exception:
                pass
    return None


@st.cache_data
def load_history():
    """Return (recent 30 daily levels, feature-baseline medians) from real data."""
    here = Path(__file__).resolve().parent
    p = here.parent / "data" / "murray_bridge_river_level_historical.csv"
    try:
        df = pd.read_csv(p, skiprows=4, names=["datetime", "water_level_m", "conductivity", "water_temp_c"])
        df["water_level_m"] = pd.to_numeric(df["water_level_m"], errors="coerce")
        df = df.dropna(subset=["water_level_m"])
        df = df[(df["water_level_m"] > -1) & (df["water_level_m"] < 6)]
        levels = df["water_level_m"].tail(30).round(3).tolist()
        d = df.copy()
        d["level_lag1"] = d["water_level_m"].shift(1)
        d["level_lag2"] = d["water_level_m"].shift(2)
        d["level_roll7"] = d["water_level_m"].shift(1).rolling(7).mean()
        d["level_change3"] = d["water_level_m"].shift(1) - d["water_level_m"].shift(4)
        d = d.dropna(subset=FEATURES)
        base = {f: round(float(d[f].median()), 3) for f in FEATURES}
        if len(levels) >= 8:
            return levels, base
    except Exception:
        pass
    return FALLBACK_LEVELS, FALLBACK_BASELINE


def coef_map(model):
    if model is not None:
        return {f: float(model.coef_[0][i]) for i, f in enumerate(FEATURES)}
    return FALLBACK_COEF


def features_from_levels(levels):
    return {
        "level_lag1": levels[-1],
        "level_lag2": levels[-2],
        "level_roll7": sum(levels[-7:]) / len(levels[-7:]),
        "level_change3": levels[-1] - levels[-4],
    }


def band_of(p):
    return "High" if p >= 0.66 else "Moderate" if p >= 0.33 else "Low"


def predict_local(model, feats):
    if model is not None:
        row = pd.DataFrame([[feats[f] for f in FEATURES]], columns=FEATURES)
        return float(model.predict_proba(row)[0][1])
    z = FALLBACK_INTERCEPT + sum(FALLBACK_COEF[f] * feats[f] for f in FEATURES)
    return 1.0 / (1.0 + math.exp(-z))


def predict(api_url, levels, model):
    feats = features_from_levels(levels)
    if api_url:
        try:
            r = requests.post(api_url.rstrip("/") + "/predict_series",
                              json={"levels": levels, "station_id": STATION["id"]}, timeout=30)
            r.raise_for_status()
            b = r.json()
            return b["flood_probability"], b["risk_band"], feats, b.get("model_source", "api")
        except Exception as exc:
            st.warning(f"API unreachable, using local model ({exc}).")
    p = predict_local(model, feats)
    return p, band_of(p), feats, "local model"


def apply_scenario(base_levels, scenario, offset):
    s = list(base_levels)
    n = len(s)
    if scenario == "Rising river":
        s = [v + (i / n) * 0.5 for i, v in enumerate(s)]
    elif scenario == "Flood watch":
        s = [v + (i / n) * 1.1 for i, v in enumerate(s)]
    return [round(v + offset, 3) for v in s]


def gauge_svg(p, color):
    r = 80
    length = math.pi * r
    off = length * (1 - p)
    return f"""
    <svg width="210" height="120" viewBox="0 0 200 118">
      <path d="M20 110 A80 80 0 0 1 180 110" fill="none" stroke="#eef2f7" stroke-width="16" stroke-linecap="round"/>
      <path d="M20 110 A80 80 0 0 1 180 110" fill="none" stroke="{color}" stroke-width="16"
            stroke-linecap="round" stroke-dasharray="{length:.1f}" stroke-dashoffset="{off:.1f}"/>
      <text x="100" y="96" text-anchor="middle" font-size="32" font-weight="700" fill="#0f2438">{round(p*100)}%</text>
      <text x="100" y="112" text-anchor="middle" font-size="11" fill="#7a8aa0">flood probability</text>
    </svg>
    """


# --------------------------------------------------------------------------
st.set_page_config(page_title="River Murray Flood Risk", page_icon="🌊", layout="wide")

model = load_model()
base_levels, baseline = load_history()
coefs = coef_map(model)
try:
    default_api = st.secrets.get("API_URL", os.getenv("API_URL", ""))
except Exception:
    default_api = os.getenv("API_URL", "")

# ---- Sidebar ----
with st.sidebar:
    st.header("Scenario")
    scenario = st.radio("River condition", ["Recent (actual)", "Rising river", "Flood watch"],
                        help="Start from the real recent series, or simulate a rising river.")
    offset = st.slider("Level offset (m)", -0.30, 1.50, 0.0, 0.05,
                       help="Shift the whole series up or down to explore thresholds.")
    st.divider()
    api_url = st.text_input("API URL (optional)", value=default_api,
                            placeholder="https://flood-risk-api.onrender.com")
    st.caption("Blank = use the model bundled in the repo.")
    with st.expander("About the model"):
        st.write(
            "Logistic Regression trained on Murray Bridge river levels. "
            f"'Flood' = level at/above the {TRAIN_RISK_THRESHOLD_M} m risk threshold "
            "(0.80 quantile). Features are lagged and rolling river levels."
        )

levels = apply_scenario(base_levels, scenario, offset)
prob, band, feats, source = predict(api_url, levels, model)

# ---- Header ----
st.title("🌊 River Murray Flood Risk — Prototype")
st.caption(
    "Next-day flood risk for Murray Bridge from the trained model. "
    "Alerts require human authorisation — this view is advisory. "
    f"Updated {datetime.now().strftime('%d %b %Y, %H:%M')}."
)

# ---- Row 1: prediction + explainability ----
c1, c2 = st.columns([1, 1.25], gap="large")

with c1:
    st.subheader("Prediction")
    st.markdown(gauge_svg(prob, BAND_HEX[band]), unsafe_allow_html=True)
    st.markdown(
        f"<div style='padding:8px 13px;border-radius:8px;color:white;background:{BAND_HEX[band]};"
        f"font-weight:600;display:inline-block'>Risk band: {band}</div>",
        unsafe_allow_html=True,
    )

    # Sensitivity band (+/- 5 cm gauge error)
    lo = predict_local(model, features_from_levels([v - 0.05 for v in levels]))
    hi = predict_local(model, features_from_levels([v + 0.05 for v in levels]))
    blo, bhi = min(lo, hi), max(lo, hi)
    st.markdown(
        f"<div style='margin-top:12px;font-size:13px;color:#4a5b70'>Sensitivity band "
        f"(&plusmn;5&nbsp;cm gauge error): <b>{blo*100:.0f}% – {bhi*100:.0f}%</b></div>",
        unsafe_allow_html=True,
    )
    st.progress(prob)

    latest = levels[-1]
    st.metric("Latest river level", f"{latest:.2f} m", f"{latest - TRAIN_RISK_THRESHOLD_M:+.2f} m vs threshold")
    st.caption(f"Source: {source} · risk threshold {TRAIN_RISK_THRESHOLD_M} m")

with c2:
    st.subheader("Why this score")
    contrib = pd.DataFrame([
        {"feature": FEATURE_LABEL[f], "contribution": round(coefs[f] * (feats[f] - baseline[f]), 3)}
        for f in FEATURES
    ])
    contrib["direction"] = contrib["contribution"].apply(lambda v: "Raises risk" if v >= 0 else "Lowers risk")
    bars = alt.Chart(contrib).mark_bar().encode(
        x=alt.X("contribution:Q", title="Contribution to risk score"),
        y=alt.Y("feature:N", sort="-x", title=None),
        color=alt.Color("direction:N",
                        scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=["#d64545", "#1f6feb"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("feature:N", title="Feature"),
                 alt.Tooltip("contribution:Q", title="Contribution", format="+.3f")],
    ).properties(height=210)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#c3ccd8").encode(x="x:Q")
    st.altair_chart(zero + bars, use_container_width=True)
    st.caption("Exact per-feature contributions for logistic regression: coefficient × (feature − typical value).")

# ---- Row 2: trend chart ----
st.subheader("Recent river level vs risk threshold")
tdf = pd.DataFrame({"days ago": list(range(-len(levels) + 1, 1)), "level (m)": levels})
line = alt.Chart(tdf).mark_line(point=True, color="#1f6feb").encode(
    x=alt.X("days ago:Q", title="days (0 = latest)"),
    y=alt.Y("level (m):Q", title="river level (m)", scale=alt.Scale(zero=False)),
    tooltip=["days ago", "level (m)"],
)
threshold = alt.Chart(pd.DataFrame({"y": [TRAIN_RISK_THRESHOLD_M]})).mark_rule(
    color="#d64545", strokeDash=[6, 4]).encode(y="y:Q")
st.altair_chart((line + threshold).properties(height=240), use_container_width=True)
st.caption(f"Dashed line = {TRAIN_RISK_THRESHOLD_M} m risk threshold (0.80 quantile of historical level).")

# ---- Row 3: map ----
st.subheader("Station")
fmap = folium.Map(location=[STATION["lat"], STATION["lon"]], zoom_start=9, tiles="CartoDB positron")
folium.CircleMarker(
    location=[STATION["lat"], STATION["lon"]], radius=12,
    color=BAND_COLOR[band], fill=True, fill_color=BAND_COLOR[band], fill_opacity=0.9,
    popup=f"{STATION['name']}: {band} ({prob*100:.0f}%)", tooltip=STATION["name"],
).add_to(fmap)
st_folium(fmap, height=380, use_container_width=True)

st.divider()
st.caption("Prototype for academic assessment (ITA602). Not for operational flood-warning use.")
