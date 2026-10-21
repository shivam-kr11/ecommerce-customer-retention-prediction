#=======================
#       Import Libraries
#=======================

import html
import json
from datetime import datetime

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
PAGES = ["Dashboard", "Customer Prediction", "Prediction History", "Model Insights"]

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


PERSONAL_KEYS = ["customer_id", "customer_name", "customer_phone", "customer_email"]

DEFAULTS = {
    "customer_id": "", "customer_name": "", "customer_phone": "", "customer_email": "",
    "age": 35, "customer_tier": "Silver", "loyalty_score": 0.371,
    "avg_review_score": 3.9663, "credit_limit": 13128.0, "signup_month": 10,
    "customer_tenure_days": 346, "country_code": "IN", "region": "region_33",
    "total_sessions": 13, "avg_session_duration": 375.5337, "total_pages_viewed": 79,
    "avg_pages_viewed": 6.0769, "total_cart_additions": 23, "avg_cart_additions": 1.7692,
    "bounce_rate": 0.7692, "recency_session_days": 25,
    "total_transactions": 2, "total_spent": 284.05, "avg_order_value": 142.025,
    "max_order_value": 176.88, "total_items": 8, "avg_items_per_order": 4.0,
    "discount_trans_ratio": 0.5, "recency_trans_days": 535,
    "cart_to_order_ratio": 0.0833333, "spend_per_session": 20.2893,
    "spend_to_credit_ratio": 0.0216353,
    "email_open_rate": 0.2303, "discount_usage_rate": 0.1270,
    "avg_campaign_budget": 200225.9231, "total_campaign_budget": 2602937.0,
    "primary_campaign_type": "Social",
    "geo_ip_region": "region_64", "region_tier": "Tier1",
    "average_income": 62187.8873, "urban_ratio": 0.5204, "internet_penetration": 0.5623,
}

FEATURE_KEYS = [key for key in DEFAULTS if key not in PERSONAL_KEYS]
STEPS = ["Personal Info", "Profile", "Website", "Purchases", "Marketing", "Region"]

if "form" not in st.session_state:
    st.session_state.form = dict(DEFAULTS)
if "step" not in st.session_state:
    st.session_state.step = 0


def reset_form():
    """Button callback: start a fresh customer."""
    st.session_state.form = dict(DEFAULTS)
    st.session_state.step = 0
    for key in list(st.session_state.keys()):
        if key.startswith("w_"):
            del st.session_state[key]


def jump_to(step_number):
    """Step button callback: save the current inputs, then open that step."""
    for key in DEFAULTS:
        widget_key = f"w_{key}"
        if widget_key in st.session_state:
            st.session_state.form[key] = st.session_state[widget_key]
    st.session_state.step = step_number


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

if page == "Dashboard":

    st.subheader("Overview")

    history_df = pd.DataFrame(st.session_state.history)
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
            "<p>Enter customer details.</p></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Open Prediction", use_container_width=True,
            on_click=go_to, args=("Customer Prediction",),
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
        recent = history_df[
            ["timestamp", "customer_id", "customer_name", "customer_status",
             "probability_class_1", "customer_tier", "total_spent"]
        ].iloc[::-1].head(5)

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
            column_config={
                "probability_class_1": st.column_config.ProgressColumn(
                    "Class 1 Probability", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )


#=======================
#       Customer Prediction Page
#=======================

elif page == "Customer Prediction":

    st.subheader("Customer Prediction")
    st.caption("Fill each step, then generate the prediction on the last one.")

    form = st.session_state.form
    step = st.session_state.step
    last_step = len(STEPS) - 1

    # Widgets that save their value into `form`, so nothing is lost between steps.
    def number(key, label, **options):
        form[key] = st.number_input(label, value=form[key], key=f"w_{key}", **options)

    def text(key, label, **options):
        form[key] = st.text_input(label, value=form[key], key=f"w_{key}", **options)

    def choice(key, label, options):
        form[key] = st.selectbox(
            label, options, index=options.index(form[key]), key=f"w_{key}"
        )

    #=======================
    #       Step Tracker
    #=======================

    with st.container(key="tracker"):
        step_cols = st.columns(len(STEPS))
        for index, name in enumerate(STEPS):
            step_cols[index].button(
                f"{index + 1}. {name}",
                key=f"jump_{index}",
                type="primary" if index == step else "secondary",
                use_container_width=True,
                on_click=jump_to,
                args=(index,),
            )

    st.caption(f"Step {step + 1} of {len(STEPS)}. Click any step to jump to it.")

    #=======================
    #       Current Step Inputs
    #=======================

    with st.container(border=True):

        st.markdown(f"#### {STEPS[step]}")

        if step == 0:
            st.caption("For your records only. Not used in the prediction.")
            c1, c2 = st.columns(2)
            with c1:
                text("customer_id", "Customer ID", placeholder="e.g. CUST-1001",
                     help="Leave empty to create one automatically.")
                text("customer_phone", "Phone Number", placeholder="e.g. 98765 43210")
            with c2:
                text("customer_name", "Full Name", placeholder="e.g. Aman Singh")
                text("customer_email", "Email", placeholder="e.g. name@email.com")

        elif step == 1:
            c1, c2, c3 = st.columns(3)
            with c1:
                number("age", "Age", min_value=18, max_value=100)
                choice("customer_tier", "Customer Tier",
                       ["Silver", "Bronze", "Gold", "Platinum"])
                number("loyalty_score", "Loyalty Score", min_value=0.0, max_value=1.0,
                       format="%.4f", help="Between 0 and 1.")
            with c2:
                number("avg_review_score", "Average Review Score",
                       min_value=0.0, max_value=5.0, help="Between 0 and 5.")
                number("credit_limit", "Credit Limit", min_value=0.0)
                number("signup_month", "Signup Month", min_value=1, max_value=12)
            with c3:
                number("customer_tenure_days", "Tenure (Days)", min_value=0)
                text("country_code", "Country Code")
                text("region", "Region")

        elif step == 2:
            c1, c2, c3 = st.columns(3)
            with c1:
                number("total_sessions", "Total Sessions", min_value=0)
                number("avg_session_duration", "Avg. Session Duration", min_value=0.0)
                number("total_pages_viewed", "Total Pages Viewed", min_value=0)
            with c2:
                number("avg_pages_viewed", "Avg. Pages Viewed", min_value=0.0)
                number("total_cart_additions", "Total Cart Additions", min_value=0)
                number("avg_cart_additions", "Avg. Cart Additions", min_value=0.0)
            with c3:
                number("bounce_rate", "Bounce Rate", min_value=0.0, max_value=1.0,
                       help="Between 0 and 1.")
                number("recency_session_days", "Days Since Last Session", min_value=0)

        elif step == 3:
            c1, c2, c3 = st.columns(3)
            with c1:
                number("total_transactions", "Total Transactions", min_value=0)
                number("total_spent", "Total Spent", min_value=0.0)
                number("avg_order_value", "Avg. Order Value", min_value=0.0)
                number("max_order_value", "Max. Order Value", min_value=0.0)
            with c2:
                number("total_items", "Total Items", min_value=0)
                number("avg_items_per_order", "Avg. Items Per Order", min_value=0.0)
                number("discount_trans_ratio", "Discounted Order Ratio",
                       min_value=0.0, max_value=1.0, help="Between 0 and 1.")
                number("recency_trans_days", "Days Since Last Order", min_value=0)
            with c3:
                number("cart_to_order_ratio", "Cart-to-Order Ratio",
                       min_value=0.0, format="%.4f")
                number("spend_per_session", "Spend Per Session", min_value=0.0)
                number("spend_to_credit_ratio", "Spend-to-Credit Ratio",
                       min_value=0.0, format="%.4f")

        elif step == 4:
            c1, c2 = st.columns(2)
            with c1:
                number("email_open_rate", "Email Open Rate", min_value=0.0,
                       max_value=1.0, help="Between 0 and 1.")
                number("discount_usage_rate", "Discount Usage Rate", min_value=0.0,
                       max_value=1.0, help="Between 0 and 1.")
                number("avg_campaign_budget", "Avg. Campaign Budget", min_value=0.0)
            with c2:
                number("total_campaign_budget", "Total Campaign Budget", min_value=0.0)
                text("primary_campaign_type", "Primary Campaign Type")

        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                text("geo_ip_region", "Geographic Region")
                text("region_tier", "Region Tier")
            with c2:
                number("average_income", "Avg. Regional Income", min_value=0.0)
                number("urban_ratio", "Urban Ratio", min_value=0.0, max_value=1.0,
                       help="Between 0 and 1.")
            with c3:
                number("internet_penetration", "Internet Penetration",
                       min_value=0.0, max_value=1.0, help="Between 0 and 1.")

    #=======================
    #       Back / Next / Generate
    #=======================

    submitted = False
    back_col, next_col = st.columns(2)

    if step > 0:
        if back_col.button(f"← Back: {STEPS[step - 1]}", use_container_width=True):
            st.session_state.step -= 1
            st.rerun()

    if step < last_step:
        if next_col.button(
            f"Next: {STEPS[step + 1]} →", type="primary", use_container_width=True
        ):
            st.session_state.step += 1
            st.rerun()
    else:
        _, middle, _ = st.columns([1, 2, 1])
        submitted = middle.button(
            "Generate Prediction", type="primary", use_container_width=True
        )

    #=======================
    #       Prepare Features
    #=======================

    features = {key: form[key] for key in FEATURE_KEYS}

    #=======================
    #       Prediction Request
    #=======================

    if submitted:
        if not api_connected:
            st.error("The backend is offline.")
        else:
            try:
                with st.spinner("Generating prediction..."):
                    api_response = requests.post(
                        f"{API_URL}/predict",
                        json={"features": features},
                        timeout=15,
                    )

                if api_response.ok:
                    result = api_response.json()

                    if "prediction" not in result:
                        st.error("The API response has no prediction.")
                        st.json(result)
                    else:
                        record = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "prediction": int(result["prediction"]),
                            "predicted_label": result.get(
                                "predicted_label", f"Class {result['prediction']}"
                            ),
                            "probability_class_1": float(result["probability_class_1"]),
                            "threshold": float(result["threshold"]),
                            "customer_status": class_word(int(result["prediction"])),
                            "customer_id": form["customer_id"].strip()
                            or f"CUST-{datetime.now():%y%m%d%H%M%S}",
                            "customer_name": form["customer_name"].strip() or "Not provided",
                            "customer_phone": form["customer_phone"].strip() or "Not provided",
                            "customer_email": form["customer_email"].strip() or "Not provided",
                            **features,
                        }
                        st.session_state.history.append(record)
                        st.session_state.latest_prediction = record
                        st.markdown(
                            '<div class="success-wrap"><div class="success-banner">'
                            "Prediction generated successfully</div></div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.error(f"API error: {api_response.status_code}")
                    st.code(api_response.text)

            except requests.RequestException as error:
                st.error(f"Could not reach the API: {error}")

    #=======================
    #       Prediction Result
    #=======================

    latest = st.session_state.latest_prediction

    if latest is not None:
        st.divider()
        st.subheader("Result")
        st.caption(
            f"ID: {latest['customer_id']}  |  Customer: {latest['customer_name']}"
            f"  |  Phone: {latest['customer_phone']}"
        )

        color_a, color_b = class_colors(latest["prediction"])
        st.markdown(
            f'<div class="status-wrap"><div class="status-badge" '
            f'style="background: linear-gradient(120deg, {color_a}, {color_b});">'
            f'{latest["customer_status"]}</div></div>',
            unsafe_allow_html=True,
        )

        if latest["prediction"] == ATTENTION_CLASS:
            st.markdown(
                '<div class="alert-card alert-bad">'
                '<div class="alert-title">Needs attention</div>'
                '<p class="alert-text">This customer may leave. '
                "Consider a follow-up or an offer.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="alert-card alert-good">'
                '<div class="alert-title">No action needed</div>'
                '<p class="alert-text">This customer looks stable.</p></div>',
                unsafe_allow_html=True,
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Customer Status", latest["customer_status"])
        c2.metric("Class 1 Probability", f"{latest['probability_class_1']:.2%}")
        c3.metric("Threshold", f"{latest['threshold']:.2%}")

        st.progress(min(max(latest["probability_class_1"], 0.0), 1.0))
        st.caption(f"Status is shown as {class_word(0)} (Class 0) or {class_word(1)} (Class 1).")

        d1, d2, d3 = st.columns(3)
        d3.button("New Customer", on_click=reset_form, use_container_width=True)
        d1.download_button(
            "Download JSON",
            data=json.dumps(latest, indent=2, default=str),
            file_name="prediction_result.json",
            mime="application/json",
            use_container_width=True,
        )
        d2.download_button(
            "Download CSV",
            data=pd.DataFrame([latest]).to_csv(index=False),
            file_name="prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )


#=======================
#       Prediction History Page
#=======================

elif page == "Prediction History":

    st.subheader("Prediction History")
    st.caption("Saved for this session only.")

    if not st.session_state.history:
        st.info("No predictions yet.")
    else:
        history_df = pd.DataFrame(st.session_state.history)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(history_df))
        c2.metric(class_word(1), int((history_df["prediction"] == 1).sum()))
        c3.metric(class_word(0), int((history_df["prediction"] == 0).sum()))

        st.dataframe(
            history_df[
                ["timestamp", "customer_id", "customer_name", "customer_phone", "customer_status",
                 "probability_class_1", "threshold", "customer_tier", "total_spent"]
            ].iloc[::-1],
            use_container_width=True,
            hide_index=True,
            column_config={
                "probability_class_1": st.column_config.ProgressColumn(
                    "Class 1 Probability", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download History CSV",
            data=history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if d2.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.latest_prediction = None
            st.rerun()

        st.markdown("#### Probability Trend")
        trend = history_df["probability_class_1"].reset_index(drop=True)
        trend.index = trend.index + 1
        st.line_chart(trend)


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
        c1.metric("Model", str(api_info.get("model_type", "N/A")))
        c2.metric(
            "Threshold",
            f"{float(threshold):.2%}" if isinstance(threshold, (int, float)) else "N/A",
        )
        c3.metric("Numeric Features", str(api_info.get("numeric_feature_count", "N/A")))
        c4.metric(
            "Categorical Features",
            str(api_info.get("categorical_feature_count", "N/A")),
        )

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