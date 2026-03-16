import streamlit as st
import pickle
import pandas as pd
from pathlib import Path


st.set_page_config(
    page_title='Delivery Time AI',
    page_icon='🚴',
    layout='wide'
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'optimized_rf_model.pkl'
ENCODERS_PATH = BASE_DIR / 'label_encoders.pkl'

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --bg-start: #fff4dc;
        --bg-end: #dff6ff;
        --card: #ffffffcc;
        --ink: #153243;
        --muted: #4f6d7a;
        --accent-1: #ff6b35;
        --accent-2: #00a8a8;
        --accent-3: #f7b801;
    }

    .stApp {
        font-family: 'Outfit', sans-serif;
        color: var(--ink);
        background:
            radial-gradient(circle at 12% 10%, #ffd89a 0%, transparent 30%),
            radial-gradient(circle at 88% 8%, #9ee8ff 0%, transparent 32%),
            radial-gradient(circle at 80% 90%, #ffe9a8 0%, transparent 28%),
            linear-gradient(135deg, var(--bg-start), var(--bg-end));
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 20px;
        background: linear-gradient(125deg, #153243 0%, #17445a 45%, #0f7c8f 100%);
        color: #f9fcff;
        box-shadow: 0 14px 38px rgba(21, 50, 67, 0.25);
        margin-bottom: 1rem;
        animation: fadeInUp 0.8s ease;
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.5rem, 2vw, 2.4rem);
        letter-spacing: 0.02em;
        margin: 0;
    }

    .hero p {
        margin-top: 0.35rem;
        color: #d8edf7;
        font-size: 1rem;
    }

    .chip-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.85rem;
    }

    .chip {
        background: #ffffff1a;
        color: #f0fbff;
        border: 1px solid #ffffff33;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.84rem;
    }

    .section-card {
        background: var(--card);
        border: 1px solid #ffffff;
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 22px rgba(20, 45, 59, 0.08);
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.08rem;
        margin-bottom: 0.4rem;
        color: #0f4258;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f6fbff 0%, #eaf4ff 100%);
        border-right: 1px solid #d5e2ea;
    }

    [data-testid="stSidebar"] * {
        color: #12384a !important;
    }

    .stMarkdown,
    .stText,
    label,
    p,
    span {
        color: #153243;
    }

    [data-testid="stMetricLabel"] {
        color: #3d5967 !important;
    }

    [data-testid="stMetricValue"] {
        color: #163a4e !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #bfd1dc !important;
        color: #102a3a !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {
        color: #102a3a !important;
        -webkit-text-fill-color: #102a3a !important;
    }

    div[data-baseweb="select"] svg {
        fill: #355569 !important;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #ffffff;
        background: linear-gradient(95deg, var(--accent-1), #ff8f5a, var(--accent-3));
        box-shadow: 0 9px 20px rgba(255, 107, 53, 0.36);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(255, 107, 53, 0.44);
    }

    .result-card {
        margin-top: 1rem;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: linear-gradient(140deg, #e8fff6 0%, #d9efff 100%);
        border: 1px solid #c3e5f6;
        animation: fadeInUp 0.5s ease;
    }

    .result-label {
        font-size: 0.9rem;
        color: var(--muted);
        letter-spacing: 0.02em;
    }

    .result-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.4rem, 2.6vw, 2.2rem);
        color: #0a3b52;
        margin-top: 0.2rem;
        font-weight: 700;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
      <h1>Food Delivery Time Predictor</h1>
      <p>AI-powered ETA estimation using distance, traffic, weather, and courier profile.</p>
      <div class="chip-row">
        <span class="chip">Random Forest Model</span>
        <span class="chip">Real-time Inputs</span>
        <span class="chip">Smart ETA Insight</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the optimized Random Forest model
try:
    with open(MODEL_PATH, 'rb') as file:
        optimized_rf_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the LabelEncoders
try:
    with open(ENCODERS_PATH, 'rb') as file:
        label_encoders = pickle.load(file)
except Exception as e:
    st.error(f"Error loading LabelEncoders: {e}")
    st.stop()

col_left, col_right = st.columns([1.35, 1], gap='large')

with col_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Delivery Metrics</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric('Avg City Speed', '28 km/h')
    with m2:
        st.metric('Current Demand', 'High')
    with m3:
        st.metric('Model Confidence', 'Strong')

    distance_km = st.number_input('Distance (km)', min_value=0.1, max_value=100.0, value=10.0, step=0.1)
    preparation_time_min = st.number_input('Preparation Time (min)', min_value=1, max_value=60, value=20)
    courier_experience_yrs = st.number_input('Courier Experience (years)', min_value=0.0, max_value=20.0, value=2.0, step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Context Factors</div>', unsafe_allow_html=True)

    weather_options = label_encoders['Weather'].classes_
    weather_selected = st.selectbox('Weather', weather_options)

    traffic_level_options = label_encoders['Traffic_Level'].classes_
    traffic_level_selected = st.selectbox('Traffic Level', traffic_level_options)

    time_of_day_options = label_encoders['Time_of_Day'].classes_
    time_of_day_selected = st.selectbox('Time of Day', time_of_day_options)

    vehicle_type_options = label_encoders['Vehicle_Type'].classes_
    vehicle_type_selected = st.selectbox('Vehicle Type', vehicle_type_options)
    st.markdown('</div>', unsafe_allow_html=True)


if st.button('Predict Delivery Time'):
    # Preprocess categorical inputs using the loaded LabelEncoders
    weather_encoded = label_encoders['Weather'].transform([weather_selected])[0]
    traffic_level_encoded = label_encoders['Traffic_Level'].transform([traffic_level_selected])[0]
    time_of_day_encoded = label_encoders['Time_of_Day'].transform([time_of_day_selected])[0]
    vehicle_type_encoded = label_encoders['Vehicle_Type'].transform([vehicle_type_selected])[0]

    # Create a DataFrame for prediction, ensuring correct column order and names
    input_data = pd.DataFrame([[distance_km, weather_encoded, traffic_level_encoded, time_of_day_encoded,
                                  vehicle_type_encoded, preparation_time_min, courier_experience_yrs]],
                                columns=['Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day',
                                           'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs'])

    # Make prediction
    prediction = optimized_rf_model.predict(input_data)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted Delivery Time</div>
            <div class="result-value">{prediction[0]:.2f} minutes</div>
        </div>
        """,
        unsafe_allow_html=True
    )
