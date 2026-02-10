import json
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Movie Hit Predictor", page_icon="🎬", layout="centered")

st.title("🎬 Movie Hit Predictor")
st.write("Predict whether a movie is likely to be a **Hit** (revenue > budget) based on simple metadata.")

# Load model + options
model = joblib.load("model.joblib")
with open("options.json", "r") as f:
    opts = json.load(f)

languages = opts["languages"]
genres = opts["genres"]

st.subheader("Inputs")

budget = st.number_input("Budget (USD)", min_value=0.0, value=5_000_000.0, step=100_000.0)
runtime = st.number_input("Runtime (minutes)", min_value=0.0, value=110.0, step=1.0)
release_year = st.number_input("Release year", min_value=1900, max_value=2100, value=2015, step=1)

original_language = st.selectbox(
    "Original language",
    languages,
    index=languages.index("en") if "en" in languages else 0
)
main_genre = st.selectbox("Main genre", genres, index=0)

# Validation
errors = []
if budget <= 0:
    errors.append("Budget must be greater than 0.")
if runtime <= 0 or runtime > 400:
    errors.append("Runtime must be between 1 and 400 minutes.")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

if st.button("Predict"):
    X = pd.DataFrame([{
        "budget": budget,
        "runtime": runtime,
        "release_year": release_year,
        "original_language": original_language,
        "main_genre": main_genre,
    }])

    p_hit = float(model.predict_proba(X)[0][1])
    p_flop = 1.0 - p_hit

    # Clear result text
    st.markdown("### Prediction")
    st.metric("Hit probability", f"{p_hit:.2f}", help="Probability of HIT (1.0 means very likely hit)")

    # ---- Interactive Gauge ----
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_hit * 100,
        number={"suffix": "%"},
        title={"text": "Flop → Hit Gauge"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "name": "Flop-leaning"},
                {"range": [40, 60], "name": "Borderline"},
                {"range": [60, 100], "name": "Hit-leaning"},
            ],
            "threshold": {
                "line": {"width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ---- Interactive Bar (Flop vs Hit) ----
    fig_bar = go.Figure(data=[
        go.Bar(
            x=["FLOP", "HIT"],
            y=[p_flop, p_hit],
            text=[f"{p_flop:.2f}", f"{p_hit:.2f}"],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Probability=%{y:.2f}<extra></extra>"
        )
    ])
    fig_bar.update_layout(
        title="Probability Breakdown",
        yaxis=dict(range=[0, 1], title="Probability"),
        xaxis=dict(title="Outcome"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Optional: a clear final label
    label = "✅ HIT" if p_hit >= 0.5 else "❌ FLOP"
    st.markdown(f"## Result: {label}")