
#=======================
#       Import Libraries
#=======================

import streamlit as st
import requests


#=======================
#       Page Configuration
#=======================

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📊",
    layout="wide"
)


#=======================
#       Application Header
#=======================

st.title("E-commerce Customer Retention Intelligence")
st.write(
    "Analyze customer behavior and generate a prediction "
    "using the trained machine learning model."
)

API_URL = "http://127.0.0.1:8000"


#=======================
#       API Connection Status
#=======================

try:
    response = requests.get(API_URL, timeout=5)
    response.raise_for_status()
    st.success("Connected to the prediction API.")
except requests.RequestException:
    st.error(
        "Cannot connect to FastAPI. Start the backend using "
        "'python -m uvicorn app:app --reload'."
    )
    st.stop()


#=======================
#       Customer Feature Input
#=======================

st.subheader("Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        loyalty_score = st.number_input(
            "Loyalty Score", min_value=0.0, max_value=1.0, value=0.371
        )
        email_open_rate = st.number_input(
            "Email Open Rate", min_value=0.0, max_value=1.0, value=0.23
        )
        discount_usage_rate = st.number_input(
            "Discount Usage Rate", min_value=0.0, max_value=1.0, value=0.127
        )
        avg_review_score = st.number_input(
            "Average Review Score", min_value=0.0, max_value=5.0, value=3.97
        )
        credit_limit = st.number_input(
            "Credit Limit", min_value=0.0, value=13128.0
        )
        signup_month = st.number_input(
            "Signup Month", min_value=1, max_value=12, value=10
        )
        customer_tenure_days = st.number_input(
            "Customer Tenure (Days)", min_value=0, value=346
        )

    with col2:
        total_sessions = st.number_input(
            "Total Sessions", min_value=0, value=13
        )
        avg_session_duration = st.number_input(
            "Average Session Duration", min_value=0.0, value=375.53
        )
        total_pages_viewed = st.number_input(
            "Total Pages Viewed", min_value=0, value=79
        )
        avg_pages_viewed = st.number_input(
            "Average Pages Viewed", min_value=0.0, value=6.08
        )
        total_cart_additions = st.number_input(
            "Total Cart Additions", min_value=0, value=23
        )
        avg_cart_additions = st.number_input(
            "Average Cart Additions", min_value=0.0, value=1.77
        )
        bounce_rate = st.number_input(
            "Bounce Rate", min_value=0.0, max_value=1.0, value=0.77
        )
        recency_session_days = st.number_input(
            "Days Since Last Session", min_value=0, value=25
        )

    
    with col1:
        total_transactions = st.number_input(
            "Total Transactions", min_value=0, value=2
        )
        total_spent = st.number_input(
            "Total Spent", min_value=0.0, value=284.05
        )
        avg_order_value = st.number_input(
            "Average Order Value", min_value=0.0, value=142.025
        )
        max_order_value = st.number_input(
            "Maximum Order Value", min_value=0.0, value=176.88
        )
        total_items = st.number_input(
            "Total Items Purchased", min_value=0, value=8
        )
        avg_items_per_order = st.number_input(
            "Average Items Per Order", min_value=0.0, value=4.0
        )
        discount_trans_ratio = st.number_input(
            "Discounted Transaction Ratio",
            min_value=0.0, max_value=1.0, value=0.5
        )
        recency_trans_days = st.number_input(
            "Days Since Last Transaction", min_value=0, value=535
        )
        avg_campaign_budget = st.number_input(
            "Average Campaign Budget", min_value=0.0, value=200225.92
        )
        total_campaign_budget = st.number_input(
            "Total Campaign Budget", min_value=0.0, value=2602937.0
        )

    with col2:
        average_income = st.number_input(
            "Average Regional Income", min_value=0.0, value=62187.89
        )
        urban_ratio = st.number_input(
            "Urban Ratio", min_value=0.0, max_value=1.0, value=0.52
        )
        internet_penetration = st.number_input(
            "Internet Penetration", min_value=0.0, max_value=1.0, value=0.56
        )
        cart_to_order_ratio = st.number_input(
            "Cart-to-Order Ratio", min_value=0.0, value=0.0833
        )
        spend_per_session = st.number_input(
            "Spend Per Session", min_value=0.0, value=20.29
        )
        spend_to_credit_ratio = st.number_input(
            "Spend-to-Credit Ratio", min_value=0.0, value=0.0216
        )

        country_code = st.text_input("Country Code", value="IN")
        region = st.text_input("Region", value="region_33")
        customer_tier = st.selectbox(
            "Customer Tier", ["Silver", "Bronze", "Gold", "Platinum"]
        )
        geo_ip_region = st.text_input(
            "Primary Geographic Region", value="region_64"
        )
        region_tier = st.text_input("Region Tier", value="Tier1")
        primary_campaign_type = st.text_input(
            "Primary Campaign Type", value="Social"
        )

    st.caption(
        "The default values are from a sample validation customer. "
        "Replace them with meaningful customer data when testing."
    )

    submitted = st.form_submit_button("Predict Customer Class")
#=======================
#       Prepare Features and Call API
#=======================

if submitted:
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
        "primary_campaign_type": primary_campaign_type
    }

    try:
        with st.spinner("Generating prediction..."):
            api_response = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=15
            )

        if api_response.ok:
            result = api_response.json()

            st.subheader("Prediction Result")

            st.metric(
                "Predicted Class",
                result["predicted_label"]
            )

            st.metric(
                "Probability of Class 1",
                f"{result['probability_class_1']:.2%}"
            )

            st.caption(
                f"Decision threshold: {result['threshold']:.2%}"
            )

        else:
            st.error(f"API error ({api_response.status_code})")
            st.code(api_response.text)

    except requests.RequestException as error:
        st.error(f"Could not reach the prediction API: {error}")
