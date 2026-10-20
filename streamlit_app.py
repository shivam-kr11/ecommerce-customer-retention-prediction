#=======================
#       Import Libraries
#=======================

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
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"


#=======================
#       Custom Styling
#=======================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8faff, #eef2ff, #f5f3ff);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827, #312e81);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #ffffff;
    }

    .hero {
        background: linear-gradient(120deg, #312e81, #4f46e5, #7c3aed);
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 22px;
        color: white;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e0e7ff;
        margin-bottom: 0;
    }

    .section-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-left: 5px solid #6366f1;
        border-radius: 12px;
        padding: 16px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 600;
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


#=======================
#       Sidebar Navigation
#=======================

st.sidebar.markdown("# 🛍️ Retention AI")
st.sidebar.caption("E-commerce Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "APPLICATION",
    [
        "Dashboard",
        "Customer Prediction",
        "Prediction History",
        "Model Insights",
    ],
    key="navigation",
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning • Analytics")


#=======================
#       API Connection
#=======================

api_connected = False
api_info = {}

try:
    response = requests.get(API_URL, timeout=3)
    response.raise_for_status()
    api_connected = True
except requests.RequestException:
    pass

if api_connected:
    try:
        response = requests.get(
            f"{API_URL}/model-info",
            timeout=3,
        )

        if response.ok:
            api_info = response.json()

    except requests.RequestException:
        pass


#=======================
#       Application Header
#=======================

st.markdown(
    """
    <div class="hero">
        <h1>Customer Retention Intelligence</h1>
        <p>
            Explore customer behaviour, generate predictions,
            and review machine learning insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_col, section_col = st.columns([1, 3])

with status_col:
    if api_connected:
        st.success("API Connected")
    else:
        st.error("API Offline")

with section_col:
    st.caption(f"Current section: {page}")

if not api_connected:
    st.warning(
        "Start the FastAPI backend in another terminal using: "
        "`python -m uvicorn app:app --reload`"
    )


#=======================
#       Dashboard Page
#=======================

if page == "Dashboard":

    st.subheader("Dashboard Overview")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    total_predictions = len(history_df)

    if total_predictions:
        class_1_count = int(
            (history_df["prediction"] == 1).sum()
        )

        class_0_count = int(
            (history_df["prediction"] == 0).sum()
        )

        average_probability = history_df[
            "probability_class_1"
        ].mean()

    else:
        class_1_count = 0
        class_0_count = 0
        average_probability = None

    #=======================
    #       KPI Cards
    #=======================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Predictions",
        total_predictions,
    )

    col2.metric(
        "Class 1 Results",
        class_1_count,
    )

    col3.metric(
        "Class 0 Results",
        class_0_count,
    )

    col4.metric(
        "Average Class 1 Probability",
        (
            f"{average_probability:.1%}"
            if average_probability is not None
            else "—"
        ),
    )

    st.divider()

    #=======================
    #       Quick Actions
    #=======================

    st.subheader("Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h3>🔍 Customer Prediction</h3>
                <p>
                    Enter customer information and generate
                    a prediction from the trained model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Customer Prediction",
            use_container_width=True,
        ):
            st.session_state.navigation = "Customer Prediction"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h3>📊 Prediction History</h3>
                <p>
                    Review previous predictions and export
                    your results for further analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "View Prediction History",
            use_container_width=True,
        ):
            st.session_state.navigation = "Prediction History"
            st.rerun()

    #=======================
    #       Recent Predictions
    #=======================

    st.subheader("Recent Predictions")

    if history_df.empty:

        st.info(
            "No predictions yet. Open Customer Prediction "
            "to generate your first result."
        )

    else:

        recent_df = history_df[
            [
                "timestamp",
                "predicted_label",
                "probability_class_1",
                "customer_tier",
                "total_spent",
            ]
        ].iloc[::-1].head(5).copy()

        recent_df["probability_class_1"] = (
            recent_df["probability_class_1"]
            .map(lambda value: f"{value:.2%}")
        )

        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True,
        )


#=======================
#       Customer Prediction Page
#=======================

elif page == "Customer Prediction":

    st.subheader("Customer Prediction")

    st.write(
        "Enter customer details below. The default values "
        "are sample values for testing."
    )

    if not api_connected:
        st.warning(
            "The API is offline. Start the backend before "
            "submitting a prediction."
        )

    #=======================
    #       Customer Input Form
    #=======================

    with st.form("customer_form"):

        st.markdown("### 1. Customer Profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age", 18, 100, 35
            )

            customer_tier = st.selectbox(
                "Customer Tier",
                ["Silver", "Bronze", "Gold", "Platinum"],
            )

            loyalty_score = st.number_input(
                "Loyalty Score",
                0.0, 1.0, 0.371,
                format="%.4f",
            )

        with c2:
            avg_review_score = st.number_input(
                "Average Review Score",
                0.0, 5.0, 3.9663,
            )

            credit_limit = st.number_input(
                "Credit Limit",
                min_value=0.0,
                value=13128.0,
            )

            signup_month = st.number_input(
                "Signup Month",
                1, 12, 10,
            )

        with c3:
            customer_tenure_days = st.number_input(
                "Customer Tenure (Days)",
                min_value=0,
                value=346,
            )

            country_code = st.text_input(
                "Country Code",
                value="IN",
            )

            region = st.text_input(
                "Region",
                value="region_33",
            )

        st.divider()

        #=======================
        #       Website Engagement
        #=======================

        st.markdown("### 2. Website Engagement")

        c1, c2, c3 = st.columns(3)

        with c1:
            total_sessions = st.number_input(
                "Total Sessions", 0, value=13
            )

            avg_session_duration = st.number_input(
                "Average Session Duration",
                min_value=0.0,
                value=375.5337,
            )

            total_pages_viewed = st.number_input(
                "Total Pages Viewed",
                0, value=79,
            )

        with c2:
            avg_pages_viewed = st.number_input(
                "Average Pages Viewed",
                min_value=0.0,
                value=6.0769,
            )

            total_cart_additions = st.number_input(
                "Total Cart Additions",
                0, value=23,
            )

            avg_cart_additions = st.number_input(
                "Average Cart Additions",
                min_value=0.0,
                value=1.7692,
            )

        with c3:
            bounce_rate = st.number_input(
                "Bounce Rate",
                0.0, 1.0, 0.7692,
            )

            recency_session_days = st.number_input(
                "Days Since Last Session",
                0, value=25,
            )

        st.divider()

        #=======================
        #       Transaction Features
        #=======================

        st.markdown("### 3. Transactions & Spending")

        c1, c2, c3 = st.columns(3)

        with c1:
            total_transactions = st.number_input(
                "Total Transactions",
                0, value=2,
            )

            total_spent = st.number_input(
                "Total Spent",
                min_value=0.0,
                value=284.05,
            )

            avg_order_value = st.number_input(
                "Average Order Value",
                min_value=0.0,
                value=142.025,
            )

            max_order_value = st.number_input(
                "Maximum Order Value",
                min_value=0.0,
                value=176.88,
            )

        with c2:
            total_items = st.number_input(
                "Total Items Purchased",
                0, value=8,
            )

            avg_items_per_order = st.number_input(
                "Average Items Per Order",
                min_value=0.0,
                value=4.0,
            )

            discount_trans_ratio = st.number_input(
                "Discounted Transaction Ratio",
                0.0, 1.0, 0.5,
            )

            recency_trans_days = st.number_input(
                "Days Since Last Transaction",
                0, value=535,
            )

        with c3:
            cart_to_order_ratio = st.number_input(
                "Cart-to-Order Ratio",
                min_value=0.0,
                value=0.0833333,
            )

            spend_per_session = st.number_input(
                "Spend Per Session",
                min_value=0.0,
                value=20.2893,
            )

            spend_to_credit_ratio = st.number_input(
                "Spend-to-Credit Ratio",
                min_value=0.0,
                value=0.0216353,
            )

        st.divider()

        #=======================
        #       Marketing Features
        #=======================

        st.markdown("### 4. Marketing")

        c1, c2 = st.columns(2)

        with c1:
            email_open_rate = st.number_input(
                "Email Open Rate",
                0.0, 1.0, 0.2303,
            )

            discount_usage_rate = st.number_input(
                "Discount Usage Rate",
                0.0, 1.0, 0.1270,
            )

            avg_campaign_budget = st.number_input(
                "Average Campaign Budget",
                min_value=0.0,
                value=200225.9231,
            )

        with c2:
            total_campaign_budget = st.number_input(
                "Total Campaign Budget",
                min_value=0.0,
                value=2602937.0,
            )

            primary_campaign_type = st.text_input(
                "Primary Campaign Type",
                value="Social",
            )

        st.divider()

        #=======================
        #       Regional Features
        #=======================

        st.markdown("### 5. Regional Context")

        c1, c2, c3 = st.columns(3)

        with c1:
            geo_ip_region = st.text_input(
                "Primary Geographic Region",
                value="region_64",
            )

            region_tier = st.text_input(
                "Region Tier",
                value="Tier1",
            )

        with c2:
            average_income = st.number_input(
                "Average Regional Income",
                min_value=0.0,
                value=62187.8873,
            )

            urban_ratio = st.number_input(
                "Urban Ratio",
                0.0, 1.0, 0.5204,
            )

        with c3:
            internet_penetration = st.number_input(
                "Internet Penetration",
                0.0, 1.0, 0.5623,
            )

        st.divider()

        submitted = st.form_submit_button(
            "🚀 Generate Prediction",
            type="primary",
            use_container_width=True,
        )

    #=======================
    #       Prepare Features
    #=======================

    features = {
        "age": age,
        "loyalty_score": loyalty_score,
        "email_open_rate": email_open_rate,
        "discount_usage_rate": discount_usage_rate,
        "avg_review_score": avg_review_score,
        "credit_limit": credit_limit,
        "signup_month": signup_month,
        "customer_tenure_days": customer_tenure_days,
        "total_sessions": total_sessions,
        "avg_session_duration": avg_session_duration,
        "total_pages_viewed": total_pages_viewed,
        "avg_pages_viewed": avg_pages_viewed,
        "total_cart_additions": total_cart_additions,
        "avg_cart_additions": avg_cart_additions,
        "bounce_rate": bounce_rate,
        "recency_session_days": recency_session_days,
        "total_transactions": total_transactions,
        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "max_order_value": max_order_value,
        "total_items": total_items,
        "avg_items_per_order": avg_items_per_order,
        "discount_trans_ratio": discount_trans_ratio,
        "recency_trans_days": recency_trans_days,
        "avg_campaign_budget": avg_campaign_budget,
        "total_campaign_budget": total_campaign_budget,
        "average_income": average_income,
        "urban_ratio": urban_ratio,
        "internet_penetration": internet_penetration,
        "cart_to_order_ratio": cart_to_order_ratio,
        "spend_per_session": spend_per_session,
        "spend_to_credit_ratio": spend_to_credit_ratio,
        "country_code": country_code,
        "region": region,
        "customer_tier": customer_tier,
        "geo_ip_region": geo_ip_region,
        "region_tier": region_tier,
        "primary_campaign_type": primary_campaign_type,
    }

    #=======================
    #       Prediction Request
    #=======================

    if submitted:

        if not api_connected:

            st.error(
                "Prediction failed. The FastAPI backend is offline."
            )

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
                        st.error("The API response is missing prediction data.")
                        st.json(result)

                    else:
                        prediction_record = {
                            "timestamp": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "prediction": int(result["prediction"]),
                            "predicted_label": result.get(
                                "predicted_label",
                                f"Class {result['prediction']}",
                            ),
                            "probability_class_1": float(
                                result["probability_class_1"]
                            ),
                            "threshold": float(result["threshold"]),
                            **features,
                        }

                        st.session_state.history.append(
                            prediction_record
                        )

                        st.session_state.latest_prediction = (
                            prediction_record
                        )

                        st.success("Prediction generated successfully!")

                else:

                    st.error(
                        f"API error: {api_response.status_code}"
                    )

                    st.code(api_response.text)

            except requests.RequestException as error:

                st.error(
                    f"Could not reach the prediction API: {error}"
                )

    #=======================
    #       Prediction Result
    #=======================

    latest = st.session_state.latest_prediction

    if latest is not None:

        st.divider()
        st.subheader("Latest Prediction Result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Class",
            latest["predicted_label"],
        )

        col2.metric(
            "Probability of Class 1",
            f"{latest['probability_class_1']:.2%}",
        )

        col3.metric(
            "Decision Threshold",
            f"{latest['threshold']:.2%}",
        )

        st.write("Probability of Class 1")

        st.progress(
            min(
                max(
                    latest["probability_class_1"],
                    0.0,
                ),
                1.0,
            )
        )

        # Class labels have not yet been verified against
        # authoritative target documentation.
        st.caption(
            "Class 0 and Class 1 are shown as returned by the model. "
            "Confirm the target definition before assigning business meanings."
        )

        latest_df = pd.DataFrame([latest])

        col1, col2 = st.columns(2)

        col1.download_button(
            "Download Result JSON",
            data=json.dumps(
                latest,
                indent=2,
                default=str,
            ),
            file_name="prediction_result.json",
            mime="application/json",
            use_container_width=True,
        )

        col2.download_button(
            "Download Result CSV",
            data=latest_df.to_csv(index=False),
            file_name="prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )


#=======================
#       Prediction History Page
#=======================

elif page == "Prediction History":

    st.subheader("Prediction History")

    st.caption(
        "History is stored for the current app session only."
    )

    if not st.session_state.history:

        st.info(
            "No predictions yet. Visit Customer Prediction "
            "to generate your first result."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Predictions",
            len(history_df),
        )

        col2.metric(
            "Class 1",
            int((history_df["prediction"] == 1).sum()),
        )

        col3.metric(
            "Class 0",
            int((history_df["prediction"] == 0).sum()),
        )

        st.markdown("### Prediction Records")

        display_df = history_df[
            [
                "timestamp",
                "predicted_label",
                "probability_class_1",
                "threshold",
                "customer_tier",
                "total_spent",
            ]
        ].iloc[::-1].copy()

        display_df["probability_class_1"] = (
            display_df["probability_class_1"]
            .map(lambda value: f"{value:.2%}")
        )

        display_df["threshold"] = (
            display_df["threshold"]
            .map(lambda value: f"{value:.2%}")
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)

        col1.download_button(
            "Download Full History CSV",
            data=history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if col2.button(
            "Clear History",
            use_container_width=True,
        ):

            st.session_state.history = []
            st.session_state.latest_prediction = None
            st.rerun()

        st.markdown("### Class 1 Probability Trend")

        trend_df = history_df[
            ["timestamp", "probability_class_1"]
        ].copy()

        trend_df = trend_df.reset_index(drop=True)

        trend_df.index = trend_df.index + 1

        st.line_chart(
            trend_df["probability_class_1"]
        )


#=======================
#       Model Insights Page
#=======================

elif page == "Model Insights":

    st.subheader("Model Insights")

    st.write(
        "Review metadata from the model currently loaded by FastAPI."
    )

    if not api_connected:

        st.warning(
            "Model information is unavailable because the API is offline."
        )

    elif not api_info:

        st.warning(
            "The API is connected, but no model metadata was returned."
        )

    else:

        model_name = api_info.get(
            "model_type",
            "Not provided",
        )

        threshold = api_info.get("threshold")

        numeric_count = api_info.get(
            "numeric_feature_count"
        )

        categorical_count = api_info.get(
            "categorical_feature_count"
        )

        total_count = api_info.get(
            "total_feature_count"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Model Type",
            str(model_name),
        )

        col2.metric(
            "Decision Threshold",
            (
                f"{float(threshold):.2%}"
                if isinstance(threshold, (int, float))
                else "Not provided"
            ),
        )

        col3.metric(
            "Numeric Features",
            (
                str(numeric_count)
                if numeric_count is not None
                else "Not provided"
            ),
        )

        col4.metric(
            "Categorical Features",
            (
                str(categorical_count)
                if categorical_count is not None
                else "Not provided"
            ),
        )

        if total_count is not None:

            st.metric(
                "Total Encoded Features",
                str(total_count),
            )

        st.markdown("### API Model Metadata")

        st.json(api_info)

    st.divider()

    st.markdown("### Understanding the prediction")

    st.markdown(
        """
        - **Predicted Class:** the class returned by the trained model.
        - **Probability of Class 1:** the model's estimated probability
          for Class 1.
        - **Decision Threshold:** the cutoff used to select the predicted class.
        - **Feature Counts:** the input feature counts reported by the API.
        """
    )

    st.info(
        "Model quality metrics such as precision, recall, F1-score, and ROC-AUC "
        "should be displayed here only when sourced from actual evaluation results."
    )


#=======================
#       Application Footer
#=======================

st.divider()

st.caption(
    "Customer Retention Intelligence | Machine Learning & Analytics"
)