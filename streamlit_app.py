#=======================
#       Import Libraries
#=======================

import html
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st


#=======================
#       Page Configuration
#=======================

st.set_page_config(
    page_title="Customer Retention Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"
PAGES = [
    "Dashboard",
    "Prediction",
    "Prediction History",
    "Model Insights",
]

# Result words and colors for each model class.
# CHECK THIS: swap the words if your target is the other way round
# (for example, if Class 1 means the customer stays).
CLASS_INFO = {
    0: ("Retained", "#10b981", "#06b6d4"),
    1: ("Churned", "#ef4444", "#f97316"),
}

# The class that needs attention (the customer who may leave).
ATTENTION_CLASS = 1


def class_word(class_id):
    return CLASS_INFO.get(class_id, (f"Class {class_id}", "#6366f1", "#8b5cf6"))[0]


def class_colors(class_id):
    info = CLASS_INFO.get(class_id, ("", "#6366f1", "#8b5cf6"))
    return info[1], info[2]


#=======================
#       Styling
#=======================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700&display=swap');

    .stApp { background: linear-gradient(135deg, #f8faff, #eef2ff, #f5f3ff); }

    .stApp p, .stApp label, .stApp li, .stApp input,
    .stApp textarea, .stApp button, .stApp [data-baseweb="select"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.01em;
    }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #111827, #312e81); }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: #ffffff; }

    .hero {
        background: linear-gradient(120deg, #312e81, #4f46e5, #7c3aed);
        padding: 34px 28px;
        text-align: center;
        border-radius: 18px;
        margin-bottom: 18px;
    }
    .hero h1 {
        color: #ffffff;
        margin: 0 0 6px 0;
        padding: 0;
        font-size: 2.4rem;
        font-weight: 700;
    }
    .hero p  { color: #e0e7ff; margin: 0; font-size: 1.05rem; }

    .section-card {
        background: #ffffff;
        padding: 16px 20px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
    }
    .section-card h3 { margin: 0 0 4px 0; }
    .section-card p  { margin: 0; color: #6b7280; }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 5px solid #6366f1;
        border-radius: 12px;
        padding: 14px;
    }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* Centered, attractive submit button */
    div[data-testid="stFormSubmitButton"] {
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(120deg, #4f46e5, #7c3aed, #ec4899);
        color: #ffffff;
        border: none;
        border-radius: 999px;
        padding: 0.75rem 3rem;
        font-family: 'Poppins', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        box-shadow: 0 10px 24px rgba(124, 58, 237, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(124, 58, 237, 0.45);
        color: #ffffff;
    }

    /* ---------- Sidebar ---------- */
    .brand { text-align: center; padding: 6px 0 12px 0; }
    .brand-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a5b4fc, #f0abfc);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .brand-sub {
        color: #c7d2fe;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .side-label {
        color: #a5b4fc;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 16px 0 6px 4px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 6px; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        width: 100%;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 12px;
        padding: 10px 14px;
        cursor: pointer;
        transition: all 0.18s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.16);
        transform: translateX(5px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(120deg, #6366f1, #a855f7);
        border-color: transparent;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.45);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
        display: none;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        margin: 0;
    }
    .pill {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .pill-on  { background: rgba(16, 185, 129, 0.18); border: 1px solid #10b981; color: #a7f3d0; }
    .pill-off { background: rgba(239, 68, 68, 0.18);  border: 1px solid #ef4444; color: #fecaca; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: currentColor; }
    .pill-on .dot { animation: pulse 1.6s infinite; }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70%  { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .side-stat {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .side-stat b {
        display: block;
        font-family: 'Poppins', sans-serif;
        font-size: 1.25rem;
        color: #ffffff;
    }
    .side-stat small { color: #c7d2fe; }

    /* ---------- Attention alert ---------- */
    .alert-card {
        text-align: center;
        border-radius: 14px;
        border-left: 6px solid;
        padding: 14px 20px;
        margin: 4px 0 18px 0;
    }
    .alert-bad  { background: #fef2f2; border-color: #ef4444; color: #991b1b; }
    .alert-good { background: #ecfdf5; border-color: #10b981; color: #065f46; }
    .alert-title { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 1.1rem; }
    .alert-text  { margin: 2px 0 0 0; }

    /* ---------- Clickable step tracker ---------- */
    .st-key-tracker div.stButton > button {
        background: #e5e7eb;
        color: #4b5563;
        border: none;
        border-radius: 999px;
        padding: 0.35rem 0.4rem;
        font-family: 'Poppins', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: none;
        transition: all 0.15s ease;
    }
    .st-key-tracker div.stButton > button:hover {
        background: #c7d2fe;
        color: #3730a3;
        transform: translateY(-2px);
    }
    .st-key-tracker div.stButton > button[kind="primary"],
    .st-key-tracker div.stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(120deg, #4f46e5, #7c3aed);
        color: #ffffff;
        padding: 0.35rem 0.4rem;
        font-size: 0.85rem;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
    }
    .st-key-tracker div.stButton > button[kind="primary"]:hover,
    .st-key-tracker div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        color: #ffffff;
        transform: translateY(-2px);
    }

    /* ---------- Step tracker (old chips) ---------- */
    .steps { display: flex; gap: 8px; margin: 6px 0 14px 0; flex-wrap: wrap; }
    .step {
        flex: 1;
        min-width: 110px;
        text-align: center;
        padding: 9px 10px;
        border-radius: 999px;
        font-family: 'Poppins', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        background: #e5e7eb;
        color: #6b7280;
    }
    .step span {
        display: inline-block;
        width: 22px;
        height: 22px;
        line-height: 22px;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.08);
        margin-right: 6px;
        font-size: 0.75rem;
    }
    .step.done   { background: #c7d2fe; color: #3730a3; }
    .step.active {
        background: linear-gradient(120deg, #4f46e5, #7c3aed);
        color: #ffffff;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
    }
    .step.active span { background: rgba(255, 255, 255, 0.25); }

    /* Result status badge */
    .status-wrap { display: flex; justify-content: center; margin: 8px 0 18px 0; }
    .status-badge {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        padding: 14px 56px;
        border-radius: 999px;
        letter-spacing: 0.02em;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
    }

    /* Centered success banner */
    .success-wrap { display: flex; justify-content: center; margin: 16px 0; }
    .success-banner {
        background: linear-gradient(120deg, #10b981, #06b6d4);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 14px 40px;
        border-radius: 999px;
        box-shadow: 0 10px 24px rgba(16, 185, 129, 0.35);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


#=======================
#       Session State
#=======================

if "history" not in st.session_state:
    st.session_state.history = []
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
if "navigation" not in st.session_state:
    st.session_state.navigation = PAGES[0]


PERSONAL_DEFAULTS = {
    "customer_id": "",
    "customer_name": "",
    "customer_phone": "",
    "customer_email": "",
}

FEATURE_DEFAULTS = {
    "age": 35,
    "loyalty_score": 0.371,
    "discount_usage_rate": 0.127,
    "email_open_rate": 0.2303,
    "avg_cart_additions": 1.7692,
    "avg_review_score": 3.9663,
    "spend_to_credit_ratio": 0.0216353,
    "region": "region_33",
}

FEATURE_CONTROLS = {
    "age": {"label": "Customer age", "min_value": 18, "max_value": 100,
            "step": 1, "format": "%d", "help": "Age in years."},
    "loyalty_score": {"label": "Loyalty score", "min_value": 0.0,
                       "max_value": 1.0, "step": 0.01, "format": "%.3f",
                       "help": "Score from 0 to 1."},
    "discount_usage_rate": {"label": "Discount usage rate", "min_value": 0.0,
                             "max_value": 1.0, "step": 0.01, "format": "%.3f",
                             "help": "Share of purchases using a discount."},
    "email_open_rate": {"label": "Email open rate", "min_value": 0.0,
                         "max_value": 1.0, "step": 0.01, "format": "%.3f",
                         "help": "Share of marketing emails opened."},
    "avg_cart_additions": {"label": "Average cart additions per session",
                            "min_value": 0.0, "step": 0.1, "format": "%.2f",
                            "help": "Average items added to cart in a session."},
    "avg_review_score": {"label": "Average review score", "min_value": 0.0,
                          "max_value": 5.0, "step": 0.1, "format": "%.2f",
                          "help": "Average rating from 0 to 5."},
    "spend_to_credit_ratio": {"label": "Spend-to-credit ratio", "min_value": 0.0,
                               "step": 0.01, "format": "%.4f",
                               "help": "Spend divided by credit limit plus one."},
    "region": {"label": "Customer region", "help": "Choose the customer's profile region."},
}

DEFAULTS = {**PERSONAL_DEFAULTS, **FEATURE_DEFAULTS}
FEATURE_KEYS = list(FEATURE_DEFAULTS)

if "form" not in st.session_state:
    st.session_state.form = dict(DEFAULTS)


def reset_form():
    """Button callback: start a fresh customer."""
    st.session_state.form = dict(DEFAULTS)
    for key in list(st.session_state.keys()):
        if key.startswith("prediction_"):
            del st.session_state[key]
    st.session_state.latest_prediction = None


def go_to(page_name):
    """Button callback: switch page safely."""
    st.session_state.navigation = page_name


#=======================
#       Sidebar Navigation
#=======================

st.sidebar.markdown(
    '<div class="brand"><div class="brand-title">Retention AI</div>'
    '<div class="brand-sub">E-commerce Intelligence</div></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Menu", PAGES, key="navigation", label_visibility="collapsed"
)


#=======================
#       API Connection
#=======================

api_connected = False
api_info = {}

try:
    requests.get(API_URL, timeout=3).raise_for_status()
    api_connected = True
except requests.RequestException:
    pass

if api_connected:
    try:
        info_response = requests.get(f"{API_URL}/model-info", timeout=3)
        if info_response.ok:
            api_info = info_response.json()
    except requests.RequestException:
        pass

MODEL_FEATURES = api_info.get("selected_input_columns", FEATURE_KEYS)
MODEL_CATEGORICAL_FEATURES = api_info.get("categorical_columns", ["region"])


@st.cache_data
def load_region_options():
    """Load known customer regions for the categorical input control."""
    region_file = Path(__file__).resolve().parent / "data" / "processed" / "customers.csv"
    try:
        regions = pd.read_csv(region_file, usecols=["region"], dtype="string")["region"]
        values = sorted(regions.dropna().unique().tolist())
        if values:
            return values
    except (OSError, ValueError, pd.errors.ParserError):
        pass
    return [f"region_{number}" for number in range(100)]


REGION_OPTIONS = load_region_options()
PREDICTION_STORE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "submissions"
    / "prediction_inputs.json"
)


def load_prediction_records():
    """Read persisted prediction details from the local JSON store."""
    if not PREDICTION_STORE_PATH.exists():
        return []
    with PREDICTION_STORE_PATH.open("r", encoding="utf-8") as json_file:
        store = json.load(json_file)
    if not isinstance(store, dict) or not isinstance(store.get("records"), list):
        raise ValueError("The prediction history JSON has an invalid structure.")
    return store["records"]


def save_prediction_record(record):
    """Append one model prediction to the local JSON store atomically."""
    records = load_prediction_records()
    records.append(record)
    write_prediction_records(records)


def write_prediction_records(records):
    """Write the complete prediction history to JSON using an atomic replace."""
    PREDICTION_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = PREDICTION_STORE_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            {"schema_version": 1, "records": records},
            json_file,
            indent=2,
            ensure_ascii=False,
        )
    temporary_path.replace(PREDICTION_STORE_PATH)


def delete_prediction_records(record_ids):
    """Remove selected prediction records and return the number removed."""
    selected_ids = set(record_ids)
    records = load_prediction_records()
    remaining_records = [
        record for record in records if record.get("record_id") not in selected_ids
    ]
    removed_count = len(records) - len(remaining_records)
    if removed_count:
        write_prediction_records(remaining_records)
    return removed_count


HISTORY_COLUMN_LABELS = {
    "record_id": "Prediction ID",
    "customer_id": "Customer ID",
    "customer_name": "Customer Name",
    "customer_phone": "Phone Number",
    "timestamp": "Predicted At",
    "model_name": "Model",
    "customer_status": "Outcome",
    "predicted_label": "Model Class",
    "probability_class_1": "Churn Probability",
    "threshold": "Decision Threshold",
    "age": "Age",
    "loyalty_score": "Loyalty Score",
    "discount_usage_rate": "Discount Usage Rate",
    "email_open_rate": "Email Open Rate",
    "avg_cart_additions": "Average Cart Additions",
    "avg_review_score": "Average Review Score",
    "spend_to_credit_ratio": "Spend-to-Credit Ratio",
    "region": "Region",
}


def history_column_config(columns):
    """Give stored prediction fields readable labels in Streamlit tables."""
    config = {
        column: HISTORY_COLUMN_LABELS.get(
            column, column.replace("_", " ").title()
        )
        for column in columns
    }
    if "probability_class_1" in columns:
        config["probability_class_1"] = st.column_config.ProgressColumn(
            "Churn Probability", min_value=0.0, max_value=100.0, format="%.1f%%"
        )
    if "threshold" in columns:
        config["threshold"] = st.column_config.NumberColumn(
            "Decision Threshold", format="%.1f%%"
        )
    numeric_formats = {
        "age": "%.0f",
        "loyalty_score": "%.3f",
        "discount_usage_rate": "%.3f",
        "email_open_rate": "%.3f",
        "avg_cart_additions": "%.2f",
        "avg_review_score": "%.2f",
        "spend_to_credit_ratio": "%.4f",
    }
    for column, number_format in numeric_formats.items():
        if column in columns:
            config[column] = st.column_config.NumberColumn(
                HISTORY_COLUMN_LABELS[column], format=number_format
            )
    return config


#=======================
#       Header
#=======================

st.markdown(
    """
    <div class="hero">
        <h1>Customer Retention Intelligence</h1>
        <p>Predict customer class from behavior and spending data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="side-label">System</div>', unsafe_allow_html=True)

if api_connected:
    st.sidebar.markdown(
        '<div class="pill pill-on"><span class="dot"></span>API connected</div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div class="pill pill-off"><span class="dot"></span>API offline</div>',
        unsafe_allow_html=True,
    )
    st.warning("Start the backend: `python -m uvicorn app:app --reload`")

stats_slot = st.sidebar.empty()


#=======================
#       Dashboard Page
#=======================

if page == "Prediction":

    st.subheader("Make a Prediction")
    st.caption("Provide only the active model's input features. Each prediction and its inputs are saved to local JSON history.")
    st.info(
        "The model uses these values to estimate customer churn. If a value comes from "
        "store activity, use the value recorded by the store rather than guessing."
    )

    with st.form("customer_data_intake_form", clear_on_submit=False):
        st.markdown("#### Customer details")
        customer_cols = st.columns(2)
        customer_name = customer_cols[0].text_input(
            "Customer name *", max_chars=120, key="prediction_customer_name"
        )
        customer_phone = customer_cols[1].text_input(
            "Phone number *", max_chars=40, key="prediction_customer_phone"
        )

        st.markdown(f"#### Model features ({len(MODEL_FEATURES)})")
        input_values = {}
        input_columns = st.columns(2)
        for index, feature in enumerate(MODEL_FEATURES):
            control = FEATURE_CONTROLS.get(
                feature,
                {"label": feature.replace("_", " ").title(), "min_value": 0.0,
                 "step": 0.01, "help": "Enter this model input."},
            )
            default_value = FEATURE_DEFAULTS.get(feature, 0.0)
            with input_columns[index % 2]:
                if feature in MODEL_CATEGORICAL_FEATURES and feature == "region":
                    region_value = (
                        default_value if default_value in REGION_OPTIONS else REGION_OPTIONS[0]
                    )
                    input_values[feature] = st.selectbox(
                        control["label"], REGION_OPTIONS,
                        index=REGION_OPTIONS.index(region_value),
                        key=f"prediction_feature_{feature}", help=control.get("help"),
                    )
                elif feature in MODEL_CATEGORICAL_FEATURES:
                    input_values[feature] = st.text_input(
                        control["label"], key=f"prediction_feature_{feature}",
                        help=control.get("help"),
                    )
                elif feature in {
                    "loyalty_score", "discount_usage_rate", "email_open_rate",
                    "avg_review_score",
                }:
                    input_values[feature] = st.slider(
                        control["label"], min_value=control["min_value"],
                        max_value=control["max_value"], value=float(default_value),
                        step=control["step"], key=f"prediction_feature_{feature}",
                        help=control.get("help"),
                    )
                else:
                    input_options = {
                        name: value for name, value in control.items()
                        if name in {"min_value", "max_value", "step", "format", "help"}
                    }
                    input_default = int(default_value) if feature == "age" else float(default_value)
                    input_values[feature] = st.number_input(
                        control["label"], value=input_default,
                        key=f"prediction_feature_{feature}", **input_options,
                    )
        submitted = st.form_submit_button(
            "Get prediction", type="primary", use_container_width=True
        )

    if submitted:
        if not customer_name.strip() or not customer_phone.strip():
            st.error("Enter the customer name and phone number before predicting.")
        elif not api_connected:
            st.error("The prediction API is offline. Start the backend, then try again.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/predict", json={"features": input_values}, timeout=20
                )
                result = response.json()
                if not response.ok or "prediction" not in result:
                    st.error(result.get("error", "The model could not score these inputs."))
                else:
                    now = datetime.now()
                    prediction = int(result["prediction"])
                    customer_id = f"CUST-{uuid4().hex[:12].upper()}"
                    record = {
                        "record_id": f"PRED-{uuid4().hex[:12].upper()}",
                        "customer_id": customer_id,
                        "customer_name": customer_name.strip(),
                        "customer_phone": customer_phone.strip(),
                        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "prediction": prediction,
                        "predicted_label": result["predicted_label"],
                        "model_name": api_info.get("model", "Retention model"),
                        "probability_class_1": float(result["probability_class_1"]),
                        "threshold": float(result["threshold"]),
                        "customer_status": class_word(prediction),
                        "saved_to_json": True,
                        **input_values,
                    }
                    try:
                        save_prediction_record(record)
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        record["saved_to_json"] = False
                        st.error(f"Prediction succeeded, but could not save its history: {error}")
                    st.session_state.history.append(record)
                    st.session_state.latest_prediction = record
            except (requests.RequestException, ValueError) as error:
                st.error(f"Could not get a prediction: {error}")

    latest = st.session_state.latest_prediction
    if latest is not None:
        st.divider()
        st.subheader("Result")
        customer_label = html.escape(latest.get("customer_name", "Customer"))
        customer_phone = html.escape(latest.get("customer_phone", "Not provided"))
        customer_id = html.escape(latest.get("customer_id", "Not provided"))
        st.caption(
            f"ID: {customer_id}  |  Customer: {customer_label}"
            f"  |  Phone: {customer_phone}"
        )
        if latest.get("saved_to_json", True):
            st.caption("Prediction saved in Prediction History.")
        else:
            st.warning("Prediction succeeded, but it could not be saved to history.")

        prediction = int(latest["prediction"])
        status = class_word(prediction)
        color_a, color_b = class_colors(prediction)
        st.markdown(
            f'<div class="status-wrap"><div class="status-badge" '
            f'style="background: linear-gradient(120deg, {color_a}, {color_b});">'
            f'{html.escape(status)}</div></div>',
            unsafe_allow_html=True,
        )

        if prediction == ATTENTION_CLASS:
            st.markdown(
                '<div class="alert-card alert-bad">'
                '<div class="alert-title">Needs attention</div>'
                '<p class="alert-text">This customer may leave. '
                'Consider a follow-up or an offer.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="alert-card alert-good">'
                '<div class="alert-title">No action needed</div>'
                '<p class="alert-text">This customer looks stable.</p></div>',
                unsafe_allow_html=True,
            )

        probability = float(latest["probability_class_1"])
        threshold = float(latest["threshold"])
        result_cols = st.columns(3)
        result_cols[0].metric("Customer Status", status)
        result_cols[1].metric("Class 1 Probability", f"{probability:.2%}")
        result_cols[2].metric("Threshold", f"{threshold:.2%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("Class 1 probability is the estimated churn probability.")

        download_cols = st.columns(3)
        download_cols[0].download_button(
            "Download JSON",
            data=json.dumps(latest, indent=2, default=str),
            file_name="prediction_result.json",
            mime="application/json",
            use_container_width=True,
        )
        download_cols[1].download_button(
            "Download CSV",
            data=pd.DataFrame([latest]).to_csv(index=False),
            file_name="prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
        download_cols[2].button(
            "New Customer", on_click=reset_form, use_container_width=True
        )

elif page == "Dashboard":

    st.subheader("Overview")

    try:
        history_df = pd.DataFrame(load_prediction_records())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        history_df = pd.DataFrame()
        st.error(f"Could not read saved prediction history: {error}")
    for column in ["customer_id", "customer_name", "customer_phone"]:
        if column not in history_df.columns:
            history_df[column] = "Not provided"
        else:
            history_df[column] = history_df[column].fillna("Not provided")
    if not history_df.empty:
        if "customer_status" not in history_df.columns:
            history_df["customer_status"] = history_df["prediction"].map(class_word)
        else:
            history_df["customer_status"] = history_df["customer_status"].fillna(
                history_df["prediction"].map(class_word)
            )
    total = len(history_df)

    if total:
        class_1 = int((history_df["prediction"] == 1).sum())
        class_0 = int((history_df["prediction"] == 0).sum())
        avg_prob = f"{history_df['probability_class_1'].mean():.1%}"
    else:
        class_1, class_0, avg_prob = 0, 0, "-"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions", total)
    c2.metric(class_word(1), class_1)
    c3.metric(class_word(0), class_0)
    c4.metric("Avg. Class 1 Probability", avg_prob)

    st.divider()
    st.subheader("Quick Actions")

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="section-card"><h3>New Prediction</h3>'
            "<p>Enter the active model's input features.</p></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Open Prediction", use_container_width=True,
            on_click=go_to, args=("Prediction",),
        )

    with right:
        st.markdown(
            '<div class="section-card"><h3>Past Results</h3>'
            "<p>Review and export predictions.</p></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Open History", use_container_width=True,
            on_click=go_to, args=("Prediction History",),
        )

    st.subheader("Recent Predictions")

    if history_df.empty:
        st.info("No predictions yet.")
    else:
        recent_columns = [column for column in [
            "timestamp", "record_id", "customer_id", "customer_name",
            "customer_phone", "customer_status",
            "probability_class_1",
        ] if column in history_df.columns] + [
            key for key in MODEL_FEATURES
            if key in history_df.columns and key not in {
                "timestamp", "record_id", "customer_id", "customer_name",
                "customer_phone", "predicted_label", "probability_class_1",
            }
        ]
        recent = history_df[recent_columns].iloc[::-1].head(5)
        recent = recent.copy()
        if "probability_class_1" in recent.columns:
            recent["probability_class_1"] *= 100

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
            column_config=history_column_config(recent.columns),
        )


#=======================
#       Prediction History Page
#=======================

elif page == "Prediction History":

    st.subheader("Prediction History")
    st.caption(f"Saved prediction inputs and results: {PREDICTION_STORE_PATH}")
    if st.session_state.pop("clear_history_selection", False):
        st.session_state.history_records_to_remove = []

    try:
        saved_records = load_prediction_records()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        saved_records = []
        st.error(f"Could not read saved predictions: {error}")

    if not saved_records:
        st.info("No saved predictions yet. Make one from the Prediction page.")
    else:
        history_df = pd.DataFrame(saved_records)
        for column in ["customer_id", "customer_name", "customer_phone"]:
            if column not in history_df.columns:
                history_df[column] = "Not provided"
            else:
                history_df[column] = history_df[column].fillna("Not provided")
        if "customer_status" not in history_df.columns:
            history_df["customer_status"] = history_df["prediction"].map(class_word)
        else:
            history_df["customer_status"] = history_df["customer_status"].fillna(
                history_df["prediction"].map(class_word)
            )
        filter_cols = st.columns(2)
        class_filter = filter_cols[0].selectbox(
            "Filter by predicted class", ["All classes", "Class 0", "Class 1"]
        )
        record_search = filter_cols[1].text_input("Search customer or record ID")

        filtered_history = history_df.copy()
        if class_filter != "All classes":
            selected_class = int(class_filter[-1])
            filtered_history = filtered_history[
                filtered_history["prediction"] == selected_class
            ]
        if record_search.strip():
            search_term = record_search.strip()
            record_ids = filtered_history.get(
                "record_id", pd.Series("", index=filtered_history.index)
            ).astype("string")
            customer_ids = filtered_history.get(
                "customer_id", pd.Series("", index=filtered_history.index)
            ).astype("string")
            customer_names = filtered_history.get(
                "customer_name", pd.Series("", index=filtered_history.index)
            ).astype("string")
            search_matches = (
                record_ids.str.contains(search_term, case=False, na=False)
                | customer_ids.str.contains(search_term, case=False, na=False)
                | customer_names.str.contains(search_term, case=False, na=False)
            )
            filtered_history = filtered_history[search_matches]

        metric_cols = st.columns(3)
        metric_cols[0].metric("Saved predictions", len(history_df))
        metric_cols[1].metric("Showing", len(filtered_history))
        metric_cols[2].metric(
            class_word(1), int((history_df["prediction"] == 1).sum())
        )

        if filtered_history.empty:
            st.info("No saved predictions match those filters.")
        else:
            display_columns = [
                column for column in [
                    "record_id", "customer_id", "customer_name", "customer_phone",
                    "timestamp", "model_name", "customer_status",
                    "probability_class_1", "threshold",
                ] if column in filtered_history.columns
            ] + [
                column for column in MODEL_FEATURES
                if column in filtered_history.columns
                and column not in {
                    "record_id", "customer_id", "customer_name", "customer_phone",
                    "timestamp", "model_name", "customer_status",
                    "probability_class_1", "threshold",
                }
            ]
            display_history = filtered_history[display_columns].iloc[::-1].copy()
            display_history["probability_class_1"] *= 100
            if "threshold" in display_history.columns:
                display_history["threshold"] *= 100
            st.dataframe(
                display_history,
                use_container_width=True,
                hide_index=True,
                column_config=history_column_config(display_history.columns),
            )

            download_cols = st.columns(2)
            download_cols[0].download_button(
                "Download filtered CSV",
                data=filtered_history.to_csv(index=False),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
            download_cols[1].download_button(
                "Download filtered JSON",
                data=json.dumps(
                    filtered_history.to_dict(orient="records"), indent=2, default=str
                ),
                file_name="prediction_history.json",
                mime="application/json",
                use_container_width=True,
            )

            st.markdown("#### Class 1 Probability Trend")
            st.line_chart(
                filtered_history["probability_class_1"].reset_index(drop=True)
            )

            st.divider()
            st.markdown("#### Remove saved predictions")
            st.caption("Choose one or more records from the current filter results.")
            record_labels = {
                record["record_id"]: (
                    f"{record['record_id']} · "
                    f"{record.get('customer_name', 'Not provided')} · "
                    f"{record.get('customer_status', class_word(record['prediction']))}"
                )
                for record in filtered_history.to_dict(orient="records")
                if record.get("record_id")
            }
            selected_record_ids = st.multiselect(
                "Select records to remove",
                options=list(record_labels),
                format_func=lambda record_id: record_labels[record_id],
                key="history_records_to_remove",
            )
            if st.button(
                "Remove selected records",
                disabled=not selected_record_ids,
                key="request_history_delete",
            ):
                st.session_state.pending_delete_ids = selected_record_ids

            pending_delete_ids = st.session_state.get("pending_delete_ids", [])
            if pending_delete_ids:
                st.warning(
                    f"This will permanently remove {len(pending_delete_ids)} "
                    "prediction record(s) from the JSON history."
                )
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button(
                    "Confirm permanent removal", type="primary", key="confirm_history_delete"
                ):
                    try:
                        removed_count = delete_prediction_records(pending_delete_ids)
                        removed_set = set(pending_delete_ids)
                        st.session_state.history = [
                            record for record in st.session_state.history
                            if record.get("record_id") not in removed_set
                        ]
                        latest_record = st.session_state.latest_prediction
                        if latest_record and latest_record.get("record_id") in removed_set:
                            st.session_state.latest_prediction = None
                        st.session_state.pending_delete_ids = []
                        st.session_state.clear_history_selection = True
                        st.success(f"Removed {removed_count} prediction record(s).")
                        st.rerun()
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        st.error(f"Could not remove the selected records: {error}")
                if cancel_col.button("Cancel", key="cancel_history_delete"):
                    st.session_state.pending_delete_ids = []
                    st.rerun()


#=======================
#       Model Insights Page
#=======================

elif page == "Model Insights":

    st.subheader("Model Insights")

    if not api_connected:
        st.warning("API offline. No model details available.")
    elif not api_info:
        st.warning("No model details returned by the API.")
    else:
        threshold = api_info.get("threshold")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", str(api_info.get("model", "N/A")))
        c2.metric(
            "Threshold",
            f"{float(threshold):.2%}" if isinstance(threshold, (int, float)) else "N/A",
        )
        c3.metric("Numeric Features", str(api_info.get("numeric_feature_count", "N/A")))
        c4.metric(
            "Categorical Features",
            str(api_info.get("categorical_feature_count", "N/A")),
        )

        st.markdown("#### Active model inputs")
        st.caption("These are the only customer attributes sent to the prediction API.")
        st.write(api_info.get("selected_input_columns", MODEL_FEATURES))

        with st.expander("Full model details"):
            st.json(api_info)

    st.divider()
    st.markdown("#### Quick Guide")
    st.markdown(
        "- **Predicted Class:** model output\n"
        "- **Class 1 Probability:** model confidence\n"
        "- **Threshold:** cutoff for Class 1"
    )


#=======================
#       Footer
#=======================

st.divider()
st.caption("Customer Retention Intelligence")


#=======================
#       Sidebar Session Stats
#=======================

with stats_slot.container():
    st.markdown('<div class="side-label">This Session</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="side-stat"><b>{len(st.session_state.history)}</b>'
        "<small>Predictions made</small></div>",
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        last = st.session_state.history[-1]
        st.markdown(
            f'<div class="side-stat"><b>{html.escape(last["customer_status"])}</b>'
            f'<small>Last: {html.escape(last["customer_name"])}</small></div>',
            unsafe_allow_html=True,
        )
