import json
import joblib
import pandas as pd
import streamlit as st

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

    pred = int(model.predict(X)[0])
    label = "✅ HIT" if pred == 1 else "❌ FLOP"
    st.markdown(f"### Result: {label}")

    if hasattr(model, "predict_proba"):
        p_hit = float(model.predict_proba(X)[0][1])
        p_flop = 1.0 - p_hit

        st.write(f"**Hit probability:** {p_hit:.2f}")
        st.write(f"**Flop probability:** {p_flop:.2f}")

        st.markdown("#### Flop → Hit Probability")
        # Progress bar expects 0–1
        st.progress(p_hit)

        # Labels under the bar
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            st.caption("FLOP (0.0)")
        with c2:
            st.caption(f"Current: {p_hit:.2f}")
        with c3:
            st.caption("HIT (1.0)")

        # Optional interpretation bands
        if p_hit < 0.4:
            st.info("Model confidence: leaning **Flop**.")
        elif p_hit < 0.6:
            st.warning("Model confidence: **borderline** (uncertain).")
        else:
            st.success("Model confidence: leaning **Hit**.")
