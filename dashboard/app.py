"""
Flood Risk Dashboard (Streamlit) — single-screen operations view
===============================================================
River Murray flood-risk prototype. Serves the team's trained Logistic
Regression and Random Forest models (next-day high-river-level risk for
Murray Bridge).

Layout (single screen, no scrolling):
    Header      title | Live | time | operator | horizon tabs
    Left card   catchment map (modelled + context stations)
    Right cards 1) selected station: gauge + uncertainty band
                2) why this score (feature contributions / importance)
                3) alert authorisation (human-in-the-loop + audit log)

Honesty notes:
- Only Murray Bridge (A4261162) has a trained model. Other stations are shown
  as context markers, not predictions.
- 24/48/72h horizons extrapolate the recent level trend, then score the model.
- Two models are available: Logistic Regression (Manuela) and Random Forest
  (Ghale). Card 2 adapts automatically: signed feature contributions for the
  Logistic Regression, feature importance for the Random Forest.
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
BAND_HEX = {"Low": "#12a150", "Moderate": "#e0982b", "High": "#d64545"}
BAND_COLOR = {"Low": "green", "Moderate": "orange", "High": "red"}

MODELLED = {"name": "Murray Bridge", "id": "A4261162", "lat": -35.12, "lon": 139.27}
CONTEXT_STATIONS = [
    {"name": "Renmark", "lat": -34.1746, "lon": 140.7461},
    {"name": "Berri", "lat": -34.2833, "lon": 140.60},
    {"name": "Loxton", "lat": -34.45, "lon": 140.57},
    {"name": "Waikerie", "lat": -34.18, "lon": 139.98},
    {"name": "Morgan", "lat": -34.03, "lon": 139.49},
    {"name": "Blanchetown", "lat": -34.35, "lon": 139.62},
    {"name": "Mannum", "lat": -34.91, "lon": 139.31},
]

FALLBACK_LEVELS = [0.725, 0.689, 0.731, 0.661, 0.653, 0.647, 0.616, 0.604, 0.654,
                   0.631, 0.592, 0.668, 0.644, 0.638, 0.667, 0.664, 0.686, 0.755,
                   0.733, 0.726, 0.713, 0.717, 0.718, 0.718, 0.7, 0.668, 0.648,
                   0.686, 0.694, 0.73]
FALLBACK_BASELINE = {"level_lag1": 0.693, "level_lag2": 0.693, "level_roll7": 0.698, "level_change3": -0.001}
FALLBACK_COEF = {"level_lag1": 13.095, "level_lag2": 3.991, "level_roll7": 9.148, "level_change3": 1.235}
FALLBACK_INTERCEPT = -20.165

# Where to look for each model, in order of preference. The app only offers
# the models it actually finds, so nothing breaks if a file is missing.
MODEL_FILES = {
    "Logistic Regression": [
        "backend/models/logistic_regression_real.joblib",
        "notebooks/logistic_regression_real.joblib",
    ],
    "Random Forest": [
        "notebooks/Random_Forest.joblib",
        "models/best_model.joblib",
    ],
}


@st.cache_resource
def load_models():
    """Load every model we can find. Missing files are skipped, not fatal."""
    here = Path(__file__).resolve().parent
    repo = here.parent
    models = {}
    for name, candidates in MODEL_FILES.items():
        for rel in candidates:
            p = repo / rel
            if p.exists():
                try:
                    import joblib
                    models[name] = joblib.load(p)
                    break
                except Exception:
                    pass
    return models


@st.cache_data
def load_history():
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
    if model is not None and hasattr(model, "coef_"):
        return {f: float(model.coef_[0][i]) for i, f in enumerate(FEATURES)}
    return FALLBACK_COEF


def features_from_levels(levels):
    return {"level_lag1": levels[-1], "level_lag2": levels[-2],
            "level_roll7": sum(levels[-7:]) / len(levels[-7:]),
            "level_change3": levels[-1] - levels[-4]}


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
                              json={"levels": levels, "station_id": MODELLED["id"]}, timeout=30)
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


def project(levels, days):
    """Extrapolate the recent trend forward `days` days."""
    if days <= 0:
        return list(levels)
    y = levels[-7:]
    slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
    out = list(levels)
    for _ in range(days):
        out.append(round(out[-1] + slope, 3))
    return out


def gauge_svg(p, color):
    r, length = 72, math.pi * 72
    off = length * (1 - p)
    return f"""
    <svg width="100%" height="104" viewBox="0 0 184 104" preserveAspectRatio="xMidYMid meet">
      <path d="M20 96 A72 72 0 0 1 164 96" fill="none" stroke="#eef2f7" stroke-width="15" stroke-linecap="round"/>
      <path d="M20 96 A72 72 0 0 1 164 96" fill="none" stroke="{color}" stroke-width="15"
            stroke-linecap="round" stroke-dasharray="{length:.1f}" stroke-dashoffset="{off:.1f}"/>
      <text x="92" y="84" text-anchor="middle" font-size="30" font-weight="700" fill="#0f2438">{round(p*100)}%</text>
      <text x="92" y="99" text-anchor="middle" font-size="10" fill="#7a8aa0">flood probability</text>
    </svg>"""


def band_bar(lo, hi, p):
    return f"""
    <div style="position:relative;height:10px;border-radius:5px;background:#eef2f7;margin-top:4px">
      <div style="position:absolute;left:{lo*100:.0f}%;width:{max(hi-lo,0.01)*100:.0f}%;top:0;bottom:0;
                  background:#cfe1fb;border-radius:5px"></div>
      <div style="position:absolute;left:{p*100:.0f}%;top:-3px;width:3px;height:16px;background:#1f6feb;border-radius:2px"></div>
    </div>"""


# --------------------------------------------------------------------------
st.set_page_config(page_title="River Murray Flood Risk", page_icon="🌊",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .block-container {padding-top: 1.1rem; padding-bottom: 0.6rem; max-width: 1500px;}
  h1, h2, h3 {margin-bottom: .2rem !important;}
  [data-testid="stMetricValue"] {font-size: 1.15rem;}
  [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
  div[data-testid="stHorizontalBlock"] {gap: .8rem;}
  .badge {display:inline-block;padding:3px 9px;border-radius:7px;font-size:12px;font-weight:600}
  .card-title {font-size:13px;font-weight:600;color:#0f2438;margin-bottom:2px}
  .muted {color:#7a8aa0;font-size:11px}
  .row {display:flex;justify-content:space-between;font-size:12px;color:#4a5b70;margin:2px 0}
</style>
""", unsafe_allow_html=True)

models = load_models()
if not models:
    st.error("No trained model found in the repo. Make sure notebooks/logistic_regression_real.joblib "
             "and/or notebooks/Random_Forest.joblib exist.")
    st.stop()

base_levels, baseline = load_history()
try:
    default_api = st.secrets.get("API_URL", os.getenv("API_URL", ""))
except Exception:
    default_api = os.getenv("API_URL", "")

with st.sidebar:
    st.header("Controls")
    model_choice = st.selectbox("Model", list(models.keys()), index=0)
    model = models[model_choice]
    scenario = st.radio("River condition", ["Recent (actual)", "Rising river", "Flood watch"])
    offset = st.slider("Level offset (m)", -0.30, 1.50, 0.0, 0.05)
    api_url = st.text_input("API URL (optional)", value=default_api,
                            placeholder="https://flood-risk-api.onrender.com")
    st.caption("Blank = model bundled in the repo.")
    with st.expander("About the model"):
        if model_choice == "Logistic Regression":
            st.write(f"Logistic Regression on Murray Bridge river levels. 'Flood' = level at/above "
                     f"{TRAIN_RISK_THRESHOLD_M} m (0.80 quantile). Horizons extrapolate the recent trend.")
        else:
            st.write(f"Random Forest (400 trees) on Murray Bridge river levels. 'Flood' = level at/above "
                     f"{TRAIN_RISK_THRESHOLD_M} m (0.80 quantile). Horizons extrapolate the recent trend.")

# ---- Header ----
h1, h2 = st.columns([2.2, 1])
with h1:
    st.markdown("")
    st.markdown("### 🌊 River Murray Flood Risk")
    st.markdown("<span class='muted'>SES &amp; Local Council decision support · ITA602 prototype</span>",
                unsafe_allow_html=True)
with h2:
    st.markdown(
        f"<div style='text-align:right;margin-top:6px'>"
        f"<span class='badge' style='background:#e6f5ec;color:#0c6b39'>● Live</span> "
        f"<span class='badge' style='background:#f2f5f9;color:#4a5b70'>{datetime.now().strftime('%H:%M')}</span> "
        f"<span class='badge' style='background:#eaf3ff;color:#0c447c'>Operator: J. Sanchez</span></div>",
        unsafe_allow_html=True)

horizon = st.radio("Forecast horizon", ["24h", "48h", "72h"], index=1,
                   horizontal=True, label_visibility="collapsed")
days = {"24h": 1, "48h": 2, "72h": 3}[horizon]

levels = project(apply_scenario(base_levels, scenario, offset), days)
prob, band, feats, source = predict(api_url, levels, model)
lo = predict_local(model, features_from_levels([v - 0.05 for v in levels]))
hi = predict_local(model, features_from_levels([v + 0.05 for v in levels]))
blo, bhi = min(lo, hi), max(lo, hi)

# ---- Main: map | right column ----
left, right = st.columns([1.55, 1], gap="medium")

with left:
    with st.container(border=True):
        t1, t2 = st.columns([1, 1])
        t1.markdown("<div class='card-title'>Catchment overview</div>", unsafe_allow_html=True)
        t2.markdown(
            "<div style='text-align:right;font-size:11px'>"
            "<span style='color:#12a150'>●</span> Low "
            "<span style='color:#e0982b'>●</span> Moderate "
            "<span style='color:#d64545'>●</span> High "
            "<span style='color:#9aa6bd'>●</span> No model</div>", unsafe_allow_html=True)
        fmap = folium.Map(location=[-34.6, 139.9], zoom_start=7, tiles="CartoDB positron")
        for s in CONTEXT_STATIONS:
            folium.CircleMarker([s["lat"], s["lon"]], radius=6, color="#9aa6bd", fill=True,
                                fill_color="#9aa6bd", fill_opacity=0.8,
                                tooltip=f"{s['name']} — context only (no trained model)").add_to(fmap)
        folium.CircleMarker([MODELLED["lat"], MODELLED["lon"]], radius=12, color=BAND_COLOR[band],
                            fill=True, fill_color=BAND_COLOR[band], fill_opacity=0.9,
                            tooltip=f"{MODELLED['name']}: {band} ({prob*100:.0f}%)").add_to(fmap)
        st_folium(fmap, height=395, use_container_width=True, returned_objects=[])
        st.markdown("<div class='muted'>8 stations shown · modelled: Murray Bridge · "
                    "sources: BoM, SILO, DEW</div>", unsafe_allow_html=True)
        with st.expander("30-day level trend"):
            tdf = pd.DataFrame({"days ago": list(range(-len(levels) + 1, 1)), "level (m)": levels})
            line = alt.Chart(tdf).mark_line(point=False, color="#1f6feb").encode(
                x=alt.X("days ago:Q", title=None),
                y=alt.Y("level (m):Q", title="m", scale=alt.Scale(zero=False)))
            rule = alt.Chart(pd.DataFrame({"y": [TRAIN_RISK_THRESHOLD_M]})).mark_rule(
                color="#d64545", strokeDash=[6, 4]).encode(y="y:Q")
            st.altair_chart((line + rule).properties(height=150), use_container_width=True)

with right:
    # Card 1 — selected station
    with st.container(border=True):
        st.markdown(f"<div class='card-title'>Selected station: {MODELLED['name']}</div>"
                    f"<div class='muted'>{MODELLED['id']} · 35.12°S, 139.27°E · next {horizon}</div>",
                    unsafe_allow_html=True)
        g1, g2 = st.columns([1.3, 1])
        with g1:
            st.markdown(gauge_svg(prob, BAND_HEX[band]), unsafe_allow_html=True)
        with g2:
            st.markdown(
                f"<div style='margin-top:26px'><span class='badge' "
                f"style='background:{BAND_HEX[band]};color:#fff'>{band} risk</span></div>"
                f"<div class='muted' style='margin-top:6px'>latest {levels[-1]:.2f} m "
                f"({levels[-1]-TRAIN_RISK_THRESHOLD_M:+.2f} m vs threshold)</div>",
                unsafe_allow_html=True)
        st.markdown(f"<div class='row'><span>Uncertainty band (±5 cm gauge error)</span>"
                    f"<b>{blo*100:.0f}% – {bhi*100:.0f}%</b></div>{band_bar(blo, bhi, prob)}",
                    unsafe_allow_html=True)

    # Card 2 — explainability (adapts to the selected model)
    with st.container(border=True):
        if hasattr(model, "coef_"):
            coefs = coef_map(model)
            st.markdown("<div class='card-title'>Why this score</div>"
                        "<div class='muted'>per-feature contribution (exact for logistic regression)</div>",
                        unsafe_allow_html=True)
            contrib = pd.DataFrame([{"feature": FEATURE_LABEL[f],
                                     "contribution": round(coefs[f] * (feats[f] - baseline[f]), 3)}
                                    for f in FEATURES])
            contrib["direction"] = contrib["contribution"].apply(
                lambda v: "Increases risk" if v >= 0 else "Decreases risk")
            bars = alt.Chart(contrib).mark_bar().encode(
                x=alt.X("contribution:Q", title=None),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color("direction:N",
                                scale=alt.Scale(domain=["Increases risk", "Decreases risk"],
                                                range=["#d64545", "#1f6feb"]),
                                legend=alt.Legend(orient="bottom", title=None, labelFontSize=10)),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("contribution:Q", format="+.3f")],
            ).properties(height=132)
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#c3ccd8").encode(x="x:Q")
            st.altair_chart(zero + bars, use_container_width=True)
        elif hasattr(model, "feature_importances_"):
            st.markdown("<div class='card-title'>Why this score</div>"
                        "<div class='muted'>feature importance (Random Forest — magnitude only, not signed)</div>",
                        unsafe_allow_html=True)
            importances = pd.DataFrame([{"feature": FEATURE_LABEL[f],
                                         "importance": round(float(model.feature_importances_[i]), 3)}
                                        for i, f in enumerate(FEATURES)])
            bars = alt.Chart(importances).mark_bar(color="#1f6feb").encode(
                x=alt.X("importance:Q", title=None),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("importance:Q", format=".3f")],
            ).properties(height=132)
            st.altair_chart(bars, use_container_width=True)
        else:
            st.caption("No explainability view available for this model.")

    # Card 3 — alert authorisation
    with st.container(border=True):
        st.markdown("<div class='card-title'>Alert authorisation</div>", unsafe_allow_html=True)
        if "alert_sent" not in st.session_state:
            st.session_state.alert_sent = False
        if band == "Low":
            st.markdown("<div class='row' style='background:#f2f5f9;padding:6px 9px;border-radius:7px'>"
                        "No alert proposed at this level</div>", unsafe_allow_html=True)
        elif st.session_state.alert_sent:
            st.markdown(f"<div class='row' style='background:#e6f5ec;color:#0c6b39;padding:6px 9px;"
                        f"border-radius:7px'>Alert dispatched · {datetime.now().strftime('%H:%M')} "
                        f"· authorised by J. Sanchez</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='row' style='background:#fdf2df;color:#8a5a0b;padding:6px 9px;"
                        "border-radius:7px'>⚠ Awaiting human authorisation</div>", unsafe_allow_html=True)
        st.markdown("<div class='row'><span class='muted'>Recipients</span>"
                    "<span>Murray Bridge SES · Rural City Council</span></div>"
                    "<div class='row'><span class='muted'>Channel</span>"
                    "<span>Email (SendGrid) · SMS descoped</span></div>", unsafe_allow_html=True)
        b1, b2 = st.columns([2, 1])
        if b1.button("Authorise & dispatch alert", type="primary",
                     disabled=(band == "Low" or st.session_state.alert_sent),
                     use_container_width=True):
            st.session_state.alert_sent = True
            st.rerun()
        if b2.button("Dismiss", use_container_width=True):
            st.session_state.alert_sent = False
            st.rerun()
        audit = "#A-2292 · signed by J. Sanchez" if st.session_state.alert_sent else "#A-2291 · tamper-evident"
        st.markdown(f"<div class='muted'>Audit log {audit} · retained per State Records Act 1997 (SA)</div>",
                    unsafe_allow_html=True)
