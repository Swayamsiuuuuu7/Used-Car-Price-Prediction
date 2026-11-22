
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# -------------------- CONFIG --------------------

st.set_page_config(
    page_title="AutoValue | Car Price Estimator",
    page_icon="🚘",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
YEAR_NOW = datetime.now().year

# -------------------- DESIGN --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #f4f7f6;
    color: #172b35;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header, footer {visibility: hidden;}

.hero {
    background: linear-gradient(120deg, #102f3b, #17645e);
    padding: 36px 40px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    color: white !important;
    font-size: 38px !important;
    margin-bottom: 8px;
    letter-spacing: -1.5px;
}

.hero p {
    color: #d1e6e2 !important;
    font-size: 15px;
    margin-bottom: 0;
}

.eyebrow {
    color: #a5e4d3;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.section-title {
    font-size: 21px;
    font-weight: 800;
    color: #173642;
    margin: 12px 0 4px;
}

.section-sub {
    color: #71828a;
    font-size: 13px;
    margin-bottom: 18px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #e3ece9;
    border-radius: 17px;
    padding: 20px;
}

label, .stMarkdown p {
    color: #425b63;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 9px;
    border-color: #dce8e4;
    background: #fbfdfc;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #167568;
    color: white;
    border: 0;
    border-radius: 11px;
    min-height: 48px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #105d53;
    color: white;
    border: 0;
}

.price-card {
    background: linear-gradient(120deg, #e2f4ed, #f2faf6);
    border: 1px solid #cce8dc;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin: 18px 0;
}

.price-label {
    color: #4d786b;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.price-value {
    color: #116653;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 8px 0;
}

.price-note {
    color: #68847a;
    font-size: 12px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e3ece9;
    padding: 16px;
    border-radius: 14px;
}

div[data-testid="stMetricValue"] {
    color: #176b5e;
}
            

/* Fix input and label visibility */

.stApp,
[data-testid="stAppViewContainer"] {
    color: #172b35 !important;
}

.stTextInput label,
.stNumberInput label,
.stSelectbox label {
    color: #344b55 !important;
    font-weight: 600 !important;
}

.stTextInput input,
.stNumberInput input {
    color: #172b35 !important;
    background: #ffffff !important;
    -webkit-text-fill-color: #172b35 !important;
}

div[data-baseweb="select"] * {
    color: #172b35 !important;
}

[data-testid="stWidgetLabel"] p {
    color: #344b55 !important;
}

input:disabled {
    -webkit-text-fill-color: #526873 !important;
    opacity: 1 !important;
}


hr {border-color: #e1eae7;}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD FILES --------------------

@st.cache_resource
def get_model():
    return joblib.load(BASE / "used_car_price_model.pkl")

@st.cache_data
def get_data():
    return pd.read_csv(BASE / "feature_engineered_car_data.csv")

try:
    model = get_model()
    data = get_data()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

# Recreate training feature
data["km_group"], km_bins = pd.cut(
    data["km_driven"], bins=20, retbins=True
)

# Dropdown data
options = {
    "name": sorted(data["name"].dropna().unique()),
    "fuel": sorted(data["fuel"].dropna().unique()),
    "seller_type": sorted(data["seller_type"].dropna().unique()),
    "transmission": sorted(data["transmission"].dropna().unique()),
    "owner": sorted(data["owner"].dropna().unique()),
    "seats": sorted(data["seats"].dropna().unique())
}

brand_map = data.groupby("name")["brand"].agg(
    lambda x: x.mode().iloc[0]
).to_dict()

# -------------------- HERO --------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">SMART AUTOMOTIVE VALUATION</div>
    <h1>Find your car's<br>realistic market value.</h1>
    <p>AI-powered used car price estimation, built with Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Vehicle configurator</div>',
            unsafe_allow_html=True)
st.markdown('<div class="section-sub">Add your vehicle details to generate an estimated resale value.</div>',
            unsafe_allow_html=True)

# -------------------- INPUT FORM --------------------

with st.form("prediction_form"):

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### 🚘 Vehicle identity")
            name = st.selectbox("Car model", options["name"])
            brand = brand_map[name]
            st.text_input("Brand", value=str(brand), disabled=True)

            year = st.number_input(
                "Manufacturing year",
                min_value=int(data["year"].min()),
                max_value=YEAR_NOW,
                value=min(2018, YEAR_NOW)
            )

            km_driven = st.number_input(
                "Kilometres driven",
                min_value=0,
                max_value=2000000,
                value=int(data["km_driven"].median()),
                step=1000
            )

            fuel = st.selectbox("Fuel type", options["fuel"])
            transmission = st.selectbox(
                "Transmission", options["transmission"]
            )

    with right:
        with st.container(border=True):
            st.markdown("### ⚙️ Technical details")

            owner = st.selectbox("Ownership", options["owner"])
            seller_type = st.selectbox(
                "Seller category", options["seller_type"]
            )

            mileage = st.number_input(
                "Mileage (km/l)",
                min_value=0.0,
                value=float(data["mileage"].median()),
                step=0.1
            )

            engine = st.number_input(
                "Engine capacity (CC)",
                min_value=0.0,
                value=float(data["engine"].median()),
                step=10.0
            )

            max_power = st.number_input(
                "Maximum power (bhp)",
                min_value=0.0,
                value=float(data["max_power"].median()),
                step=1.0
            )

            seats = st.selectbox("Seating capacity", options["seats"])

    st.markdown("")
    submitted = st.form_submit_button(
        "✦  Estimate my car's value",
        use_container_width=True
    )

# -------------------- PREDICTION --------------------

if submitted:
    km_group = pd.cut([km_driven], bins=km_bins)[0]

    input_data = pd.DataFrame([{
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "brand": brand,
        "vehicle_age": YEAR_NOW - int(year),
        "km_group": km_group
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.markdown("### Your valuation")

        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Estimated resale value</div>
            <div class="price-value">₹ {prediction:,.0f}</div>
            <div class="price-note">AI-generated estimate • Random Forest Regressor</div>
        </div>
        """, unsafe_allow_html=True)

        a, b, c = st.columns(3)
        a.metric("Vehicle age", f"{YEAR_NOW - int(year)} years")
        b.metric("Distance driven", f"{km_driven:,} km")
        c.metric("Engine", f"{engine:,.0f} CC")

        with st.expander("View prediction input details"):
            st.dataframe(input_data, use_container_width=True)

        st.caption(
            "This is a machine learning estimate, not a guaranteed "
            "market offer. Actual resale value can vary."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------- FOOTER --------------------

st.markdown("---")
st.caption(
    "AUTOVALUE  •  Used Car Price Prediction  •  "
    "Machine Learning Project"
)
