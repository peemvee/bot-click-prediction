import streamlit as st
import pandas as pd
import pickle

# 1) Load your full‐feature model (trained on synthetic_click_data_v4.xlsx)
model = pickle.load(open("bot_model.pkl", "rb"))

st.title("Bot vs Human Click Predictor")

# 2) Only the six inputs you collect from the UI:
click_freq       = st.number_input("Click frequency", min_value=1, max_value=15, value=5)
session_duration = st.number_input("Session duration (s)", min_value=1, max_value=120, value=60)
mouse_moves      = st.number_input("Mouse movements", min_value=1, max_value=300, value=100)
scroll_depth     = st.slider("Scroll depth %", 0, 100, 75)
num_pages        = st.number_input("Pages visited", min_value=1, max_value=20, value=8)
click_std        = st.number_input("Click‐interval std (s)", min_value=0.0, max_value=20.0, value=5.0)

if st.button("Predict"):
    # 3) Build a single‐row dict with *all* the features your model expects:
    row = {
        # — Label‑encoded features (just fill with 0’s so no missing columns)
        "ip_address":        0,
        "device":            0,
        "browser":           0,
        "click_time":        0,
        "session_end_time":  session_duration,  # or 0 if you prefer
        "geo_location":      0,
        "referrer":          0,

        # — Your real numeric inputs
        "click_frequency":   click_freq,
        "session_duration":  session_duration,
        "mouse_movements":   mouse_moves,
        "scroll_depth":      scroll_depth,
        "time_on_page":      session_duration,
        "num_pages_visited": num_pages,
        "click_interval_std":click_std,

        # — The two other numeric features you used in training
        "click_pattern_count": 3
    }

    # 4) Make sure we preserve the exact column order from training
    feature_order = [
        "ip_address","device","browser","click_time","session_end_time","geo_location",
        "click_frequency","session_duration","referrer","mouse_movements",
        "click_pattern_count","scroll_depth","time_on_page",
        "num_pages_visited","click_interval_std"
    ]

    df = pd.DataFrame([row], columns=feature_order)

    # 5) Predict and display
    pred = model.predict(df)[0]
    st.write("**Prediction:**", "🤖 Bot" if pred else "👤 Human")
