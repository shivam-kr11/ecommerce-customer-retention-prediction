#=======================
#       Import Libraries
#=======================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)


#=======================
#       Configuration
#=======================

path = "data/raw"


#=======================
#       Read File Names
#=======================

files = sorted(os.listdir(path))

print("\nAvailable Files in Raw Directory:")
print(files)


#=======================
#       Inspect All Raw CSV Files
#=======================

for file in files:
    if file.endswith(".csv"):
        file_path = os.path.join(path, file)
        data = pd.read_csv(file_path)

        print("\n" + "=" * 50)
        print("File Name:", file)
        print("=" * 50)
        print("Shape (Rows, Columns):", data.shape)
        print("\nColumns List:")
        print(data.columns.tolist())
        print("\nData Types:")
        print(data.dtypes)
        print("\nMissing Values Count:")
        missing_counts = data.isnull().sum()
        print(missing_counts[missing_counts > 0] if (missing_counts > 0).any() else "No missing values")
        print("\nDuplicate Rows Count:", data.duplicated().sum())
        print("\nFirst 3 Rows:")
        print(data.head(3))


#=======================
#       Inspect Target Distribution (train.csv)
#=======================

train_data = pd.read_csv(os.path.join(path, "train.csv"))

print("\n" + "=" * 50)
print("Target Variable Inspection (train.csv)")
print("=" * 50)
print("Target Class Counts:")
print(train_data["target"].value_counts())
print("\nTarget Class Proportions (%):")
print((train_data["target"].value_counts(normalize=True) * 100).round(2))


#=======================
#       Validate Key Identifiers Across Datasets
#=======================

customers = pd.read_csv(os.path.join(path, "customers.csv"))
sessions = pd.read_csv(os.path.join(path, "sessions.csv"))
transactions = pd.read_csv(os.path.join(path, "transactions.csv"))
test_data = pd.read_csv(os.path.join(path, "test.csv"))

print("\n" + "=" * 50)
print("Key Identifier Cross-Table Verification")
print("=" * 50)
print("Unique Customers in customers.csv:", customers["customer_id"].nunique())
print("Unique Customers in train.csv:    ", train_data["customer_id"].nunique())
print("Unique Customers in test.csv:     ", test_data["customer_id"].nunique())
print("Unique Customers in sessions.csv: ", sessions["customer_id"].nunique())
print("Unique Customers in transactions.csv:", transactions["customer_id"].nunique())

print("\nAre all train customer_ids present in customers.csv?:",
      train_data["customer_id"].isin(customers["customer_id"]).all())
print("Are all test customer_ids present in customers.csv?: ",
      test_data["customer_id"].isin(customers["customer_id"]).all())


#=======================
#       EDA - Target Variable Visualization
#=======================

print("\n" + "=" * 50)
print("EDA: Target Distribution Analysis")
print("=" * 50)

plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=train_data)
plt.title("Target Distribution (0 = Retained, 1 = Churned)")
plt.xlabel("Target")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


#=======================
#       EDA - Customer Numerical Features
#=======================

print("\n" + "=" * 50)
print("EDA: Customer Numerical Features Summary")
print("=" * 50)

customer_num_cols = ["loyalty_score", "email_open_rate", "discount_usage_rate", "avg_review_score"]
print(customers[customer_num_cols].describe())

customers[customer_num_cols].hist(figsize=(10, 8), bins=30)
plt.suptitle("Customer Numerical Features Distributions")
plt.tight_layout()
plt.show()


#=======================
#       EDA - Customer Categorical Features
#=======================

print("\n" + "=" * 50)
print("EDA: Customer Categorical Distributions")
print("=" * 50)

print("\nCountry Code Distribution (Top 10):")
print(customers["country_code"].value_counts().head(10))

print("\nRegion Distribution (Top 10):")
print(customers["region"].value_counts().head(10))

print("\nCustomer Tier Distribution:")
print(customers["customer_tier"].value_counts())

print("\nReferral Code Distribution (Top 10):")
print(customers["referral_code"].value_counts().head(10))


#=======================
#       EDA - Session Engagement Features
#=======================

print("\n" + "=" * 50)
print("EDA: Session Engagement Summary")
print("=" * 50)

session_num_cols = ["session_duration", "pages_viewed", "cart_additions"]
print(sessions[session_num_cols].describe())

print("\nTraffic Source Distribution:")
print(sessions["traffic_source"].value_counts())

print("\nDevice Type Distribution:")
print(sessions["device_type"].value_counts())

print("\nBounce Flag Distribution:")
print(sessions["bounce_flag"].value_counts())

sessions[session_num_cols].hist(figsize=(10, 6), bins=30)
plt.suptitle("Session Engagement Metrics Distributions")
plt.tight_layout()
plt.show()


#=======================
#       EDA - Transaction Patterns
#=======================

print("\n" + "=" * 50)
print("EDA: Transaction Behavior Summary")
print("=" * 50)

# Parsing order_value text format for exploratory numerical analysis
trans_clean_value = (
    transactions["order_value"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

print("\nCleaned Order Value Summary:")
print(trans_clean_value.describe())

print("\nItems Count Summary:")
print(transactions["items_count"].describe())

print("\nPayment Method Distribution:")
print(transactions["payment_method"].value_counts())

print("\nDiscount Applied Distribution:")
print(transactions["discount_applied"].value_counts())

print("\nShipping Speed Distribution:")
print(transactions["shipping_speed"].value_counts())

print("\nHigh Value Flag Distribution:")
print(transactions["high_value_flag"].value_counts())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(y=trans_clean_value)
plt.title("Order Value Boxplot")
plt.ylabel("Order Value ($)")

plt.subplot(1, 2, 2)
sns.boxplot(y=transactions["items_count"])
plt.title("Items Count Boxplot")
plt.ylabel("Items Count")
plt.tight_layout()
plt.show()


#=======================
#       EDA - Geographic and Marketing Signals
#=======================

geo_data = pd.read_csv(os.path.join(path, "geo_data.csv"))
marketing = pd.read_csv(os.path.join(path, "marketing_campaigns.csv"))

print("\n" + "=" * 50)
print("EDA: Geographic and Marketing Summary")
print("=" * 50)

print("\nGeographical Metrics Summary:")
print(geo_data.describe())

print("\nRegion Tier Distribution:")
print(geo_data["region_tier"].value_counts())

print("\nMarketing Campaign Type Distribution:")
print(marketing["campaign_type"].value_counts())

print("\nTarget Region Distribution (Top 10):")
print(marketing["region_target"].value_counts().head(10))

geo_data[["average_income", "urban_ratio", "internet_penetration"]].hist(
    figsize=(10, 6), bins=20
)
plt.suptitle("Geographical Metrics Distributions")
plt.tight_layout()
plt.show()


#=======================
#       Bivariate Analysis - Features vs Target
#=======================

print("\n" + "=" * 50)
print("Bivariate Analysis: Customer Features by Target Class")
print("=" * 50)

train_merged = pd.merge(customers, train_data, on="customer_id", how="inner")

print("\nAverage Features by Target (0 = Retained, 1 = Churned):")
print(train_merged.groupby("target")[
    ["loyalty_score", "discount_usage_rate", "avg_review_score", "email_open_rate"]
].mean())

print("\nMedian Features by Target (0 = Retained, 1 = Churned):")
print(train_merged.groupby("target")[
    ["loyalty_score", "discount_usage_rate", "avg_review_score", "email_open_rate"]
].median())

# Plotting key features by target class
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x="target", y="loyalty_score", data=train_merged, ax=axes[0])
axes[0].set_title("Loyalty Score by Target")
axes[0].set_xlabel("Target (0 = Retained, 1 = Churned)")
axes[0].set_ylabel("Mean Loyalty Score")

sns.barplot(x="target", y="discount_usage_rate", data=train_merged, ax=axes[1])
axes[1].set_title("Discount Usage Rate by Target")
axes[1].set_xlabel("Target (0 = Retained, 1 = Churned)")
axes[1].set_ylabel("Mean Discount Usage Rate")
plt.tight_layout()
plt.show()


#=======================
#       Missing-Value Analysis Across Tables
#=======================

print("\n" + "=" * 50)
print("Missing-Value Analysis")
print("=" * 50)

print("\nCustomers Missing Values:")
cust_missing = customers.isnull().sum()
cust_missing_pct = (cust_missing / len(customers)) * 100
missing_cust_df = pd.DataFrame({"Missing Count": cust_missing, "Percentage (%)": cust_missing_pct.round(2)})
print(missing_cust_df[missing_cust_df["Missing Count"] > 0])

print("\nSessions Missing Values:")
sess_missing = sessions.isnull().sum()
sess_missing_pct = (sess_missing / len(sessions)) * 100
missing_sess_df = pd.DataFrame({"Missing Count": sess_missing, "Percentage (%)": sess_missing_pct.round(2)})
print(missing_sess_df[missing_sess_df["Missing Count"] > 0])

print("\nTransactions Missing Values:")
trans_missing = transactions.isnull().sum()
trans_missing_pct = (trans_missing / len(transactions)) * 100
missing_trans_df = pd.DataFrame({"Missing Count": trans_missing, "Percentage (%)": trans_missing_pct.round(2)})
print(missing_trans_df[missing_trans_df["Missing Count"] > 0] if (trans_missing > 0).any() else "No missing values")


#=======================
#       Data Type Cleaning and Standardization
#=======================

print("\n" + "=" * 50)
print("Data Type Cleaning & Standardization")
print("=" * 50)

# Clean Customers age and credit_limit
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers["credit_limit"] = (
    customers["credit_limit"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)
customers["credit_limit"] = pd.to_numeric(customers["credit_limit"], errors="coerce")

# Standardize country_code categories
customers["country_code"] = customers["country_code"].replace({
    "usa": "USA",
    "U.S.": "USA"
})

# Parse Datetime columns
customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
sessions["session_timestamp"] = pd.to_datetime(sessions["session_timestamp"], errors="coerce")
transactions["transaction_timestamp"] = pd.to_datetime(transactions["transaction_timestamp"], errors="coerce")

# Clean Transactions order_value
transactions["order_value"] = (
    transactions["order_value"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)
transactions["order_value"] = pd.to_numeric(transactions["order_value"], errors="coerce")

# Clean Marketing campaign_budget
marketing["campaign_budget"] = (
    marketing["campaign_budget"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)
marketing["campaign_budget"] = pd.to_numeric(marketing["campaign_budget"], errors="coerce")
marketing["start_date"] = pd.to_datetime(marketing["start_date"], errors="coerce")
marketing["end_date"] = pd.to_datetime(marketing["end_date"], errors="coerce")

print("Cleaned Data Types:")
print("Customers (age, credit_limit):", customers["age"].dtype, customers["credit_limit"].dtype)
print("Transactions (order_value):  ", transactions["order_value"].dtype)
print("Marketing (campaign_budget): ", marketing["campaign_budget"].dtype)


#=======================
#       Outlier Detection via IQR Method
#=======================

print("\n" + "=" * 50)
print("Outlier Detection (IQR Method)")
print("=" * 50)

outlier_targets = [
    (customers, "customers", ["age", "loyalty_score", "credit_limit"]),
    (sessions, "sessions", ["session_duration", "pages_viewed", "cart_additions"]),
    (transactions, "transactions", ["order_value", "items_count"])
]

for df, df_name, cols in outlier_targets:
    print(f"\n--- Outlier Analysis for {df_name} ---")
    for col in cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
        outlier_pct = (outlier_count / len(series)) * 100
        print(f"Column: {col:<20} | Q1: {q1:<8.2f} | Q3: {q3:<8.2f} | Lower: {lower_bound:<8.2f} | Upper: {upper_bound:<8.2f} | Outliers: {outlier_count:<7} ({outlier_pct:.2f}%)")


#=======================
#       Train-Validation Customer Partition (Prevent Leakage)
#=======================

# Establish train/val split strictly at customer level before calculating any imputations or outlier caps
train_cust_ids, val_cust_ids = train_test_split(
    train_data["customer_id"],
    test_size=0.20,
    random_state=42,
    stratify=train_data["target"]
)
train_cust_set = set(train_cust_ids)
print(f"\nLeakage Prevention: Partitioned {len(train_cust_ids)} training customers and {len(val_cust_ids)} validation customers.")


#=======================
#       Missing-Value Handling & Imputation (Train Set Only)
#=======================

print("\n" + "=" * 50)
print("Missing-Value Imputation (Fit Strictly on Train Data)")
print("=" * 50)

# Impute email_open_rate with median derived strictly from training customers
median_email = customers.loc[customers["customer_id"].isin(train_cust_set), "email_open_rate"].median()
customers["email_open_rate"] = customers["email_open_rate"].fillna(median_email)

# Impute age with median derived strictly from training customers
median_age = customers.loc[customers["customer_id"].isin(train_cust_set), "age"].median()
customers["age"] = customers["age"].fillna(median_age)

# Impute credit_limit with median derived strictly from training customers
median_credit = customers.loc[customers["customer_id"].isin(train_cust_set), "credit_limit"].median()
customers["credit_limit"] = customers["credit_limit"].fillna(median_credit)

# Impute pages_viewed in sessions with median derived strictly from training customer sessions
train_sess_mask = sessions["customer_id"].isin(train_cust_set)
median_pages = sessions.loc[train_sess_mask, "pages_viewed"].median()
sessions["pages_viewed"] = sessions["pages_viewed"].fillna(median_pages)

print("Remaining Missing Values in Customers:   ", customers[["email_open_rate", "age", "credit_limit"]].isnull().sum().sum())
print("Remaining Missing Values in Sessions:    ", sessions["pages_viewed"].isnull().sum())
print("Remaining Missing Values in Transactions:", transactions["order_value"].isnull().sum())


#=======================
#       Outlier Handling (Capping / Winsorization - Train Set Only)
#=======================

print("\n" + "=" * 50)
print("Outlier Treatment (Capping Derived Strictly from Train Data)")
print("=" * 50)

# Cap order_value at 99th percentile derived strictly from training customer transactions
train_trans_mask = transactions["customer_id"].isin(train_cust_set)
order_val_p99 = transactions.loc[train_trans_mask, "order_value"].quantile(0.99)
print(f"Order Value 99th Percentile Cap (Train Only): {order_val_p99:.2f}")
print("Order Value Max Before Capping: ", transactions["order_value"].max())
transactions["order_value"] = transactions["order_value"].clip(upper=order_val_p99)
print("Order Value Max After Capping:  ", transactions["order_value"].max())

# Cap session_duration at 99th percentile derived strictly from training customer sessions
session_dur_p99 = sessions.loc[train_sess_mask, "session_duration"].quantile(0.99)
print(f"\nSession Duration 99th Percentile Cap (Train Only): {session_dur_p99:.2f}")
print("Session Duration Max Before Capping: ", sessions["session_duration"].max())
sessions["session_duration"] = sessions["session_duration"].clip(upper=session_dur_p99)
print("Session Duration Max After Capping:  ", sessions["session_duration"].max())


#=======================
#       Feature Engineering - Customer Demographics & Tenure
#=======================

print("\n" + "=" * 50)
print("Feature Engineering: Customer Profile & Tenure")
print("=" * 50)

# Derive tenure relative to the true latest date across ecosystem (eliminating negative tenure & recency)
reference_date = max(
    customers["signup_date"].max(),
    sessions["session_timestamp"].max(),
    transactions["transaction_timestamp"].max()
)
print(f"Observation Cutoff Reference Date: {reference_date.date()}")

customers["signup_year"] = customers["signup_date"].dt.year
customers["signup_month"] = customers["signup_date"].dt.month
customers["customer_tenure_days"] = (reference_date - customers["signup_date"]).dt.days

# Fill any missing tenure/date components with median derived strictly from training customers
train_cust_mask = customers["customer_id"].isin(train_cust_set)
customers["customer_tenure_days"] = customers["customer_tenure_days"].fillna(
    customers.loc[train_cust_mask, "customer_tenure_days"].median()
)
customers["signup_year"] = customers["signup_year"].fillna(
    customers.loc[train_cust_mask, "signup_year"].median()
)
customers["signup_month"] = customers["signup_month"].fillna(
    customers.loc[train_cust_mask, "signup_month"].median()
)

print("Engineered Customer Features (Non-Negative Tenures):")
print(customers[["customer_id", "signup_year", "signup_month", "customer_tenure_days"]].head(3))


#=======================
#       Feature Engineering - Session Behavioral Aggregations
#=======================

print("\n" + "=" * 50)
print("Feature Engineering: Session Behavior Aggregations")
print("=" * 50)

# Standardize bounce_flag to numeric indicator
sessions["is_bounce"] = sessions["bounce_flag"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)

# Calculate days since last session relative to reference_date
sessions["days_since_session"] = (reference_date - sessions["session_timestamp"]).dt.days

session_features = sessions.groupby("customer_id").agg(
    total_sessions=("session_id", "count"),
    avg_session_duration=("session_duration", "mean"),
    total_pages_viewed=("pages_viewed", "sum"),
    avg_pages_viewed=("pages_viewed", "mean"),
    total_cart_additions=("cart_additions", "sum"),
    avg_cart_additions=("cart_additions", "mean"),
    bounce_rate=("is_bounce", "mean"),
    recency_session_days=("days_since_session", "min")
).reset_index()

print("Session Features Shape:", session_features.shape)
print(session_features.head(3))


#=======================
#       Feature Engineering - Transaction Aggregations
#=======================

print("\n" + "=" * 50)
print("Feature Engineering: Transaction Financial Aggregations")
print("=" * 50)

# Standardize discount_applied to numeric indicator
transactions["is_discounted"] = transactions["discount_applied"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)

# Calculate days since last transaction
transactions["days_since_trans"] = (reference_date - transactions["transaction_timestamp"]).dt.days

transaction_features = transactions.groupby("customer_id").agg(
    total_transactions=("transaction_id", "count"),
    total_spent=("order_value", "sum"),
    avg_order_value=("order_value", "mean"),
    max_order_value=("order_value", "max"),
    total_items=("items_count", "sum"),
    avg_items_per_order=("items_count", "mean"),
    discount_trans_ratio=("is_discounted", "mean"),
    recency_trans_days=("days_since_trans", "min")
).reset_index()

print("Transaction Features Shape:", transaction_features.shape)
print(transaction_features.head(3))


#=======================
#       Feature Engineering - Primary Customer Region
#=======================

print("\n" + "=" * 50)
print("Feature Engineering: Primary Customer Region Mapping")
print("=" * 50)

# Resolve customer's most visited geographic region from sessions
customer_region_counts = (
    sessions.groupby(["customer_id", "geo_ip_region"])
    .size()
    .reset_index(name="region_visits")
)

customer_primary_region = (
    customer_region_counts.sort_values(["customer_id", "region_visits"], ascending=[True, False])
    .drop_duplicates(subset="customer_id")
    [["customer_id", "geo_ip_region"]]
)

print("Primary Region Mapping Shape:", customer_primary_region.shape)
print(customer_primary_region.head(3))


#=======================
#       Feature Engineering - Marketing Campaign Integration
#=======================

print("\n" + "=" * 50)
print("Feature Engineering: Marketing Campaign Integration")
print("=" * 50)

# Merge sessions with marketing campaigns on campaign_id
session_campaigns = pd.merge(
    sessions[["session_id", "customer_id", "campaign_id"]],
    marketing[["campaign_id", "campaign_type", "campaign_budget"]],
    on="campaign_id",
    how="left"
)

# Aggregate campaign budget metrics per customer
campaign_features = session_campaigns.groupby("customer_id").agg(
    avg_campaign_budget=("campaign_budget", "mean"),
    total_campaign_budget=("campaign_budget", "sum")
).reset_index()

# Determine primary campaign type driving customer engagement
customer_campaign_type = (
    session_campaigns.groupby(["customer_id", "campaign_type"])
    .size()
    .reset_index(name="type_visits")
    .sort_values(["customer_id", "type_visits"], ascending=[True, False])
    .drop_duplicates(subset="customer_id")
    [["customer_id", "campaign_type"]]
    .rename(columns={"campaign_type": "primary_campaign_type"})
)

print("Campaign Features Shape:     ", campaign_features.shape)
print("Customer Campaign Type Shape:", customer_campaign_type.shape)


#=======================
#       Data Integration - Unified Customer Master Feature Matrix
#=======================

print("\n" + "=" * 50)
print("Data Integration: Merging All Feature Tables")
print("=" * 50)

# Merge customer profiles with session, transaction, region, and marketing metrics
master_df = customers.copy()

master_df = pd.merge(master_df, session_features, on="customer_id", how="left")
master_df = pd.merge(master_df, transaction_features, on="customer_id", how="left")
master_df = pd.merge(master_df, customer_primary_region, on="customer_id", how="left")
master_df = pd.merge(master_df, campaign_features, on="customer_id", how="left")
master_df = pd.merge(master_df, customer_campaign_type, on="customer_id", how="left")

# Merge Geo-economic contextual signals using primary region
master_df = pd.merge(master_df, geo_data, on="geo_ip_region", how="left")

# Impute zero/default values for customers without marketing campaign visits
master_df["avg_campaign_budget"] = master_df["avg_campaign_budget"].fillna(0)
master_df["total_campaign_budget"] = master_df["total_campaign_budget"].fillna(0)
master_df["primary_campaign_type"] = master_df["primary_campaign_type"].fillna("None")

# Impute zero activity for customers without recorded sessions or transactions
zero_fill_cols = [
    "total_sessions", "avg_session_duration", "total_pages_viewed", "avg_pages_viewed",
    "total_cart_additions", "avg_cart_additions", "bounce_rate",
    "total_transactions", "total_spent", "avg_order_value", "max_order_value",
    "total_items", "avg_items_per_order", "discount_trans_ratio"
]
master_df[zero_fill_cols] = master_df[zero_fill_cols].fillna(0)

# Impute high default recency for customers with no sessions or transactions
master_df["recency_session_days"] = master_df["recency_session_days"].fillna(9999)
master_df["recency_trans_days"] = master_df["recency_trans_days"].fillna(9999)


#=======================
#       Cross-Domain Behavioral Interaction Ratios
#=======================

print("\n" + "=" * 50)
print("Feature Engineering: Cross-Domain Behavioral Ratios")
print("=" * 50)

# Cart-to-purchase conversion efficiency
master_df["cart_to_order_ratio"] = master_df["total_transactions"] / (master_df["total_cart_additions"] + 1)

# Monetary spend velocity per session
master_df["spend_per_session"] = master_df["total_spent"] / (master_df["total_sessions"] + 1)

# Page engagement intensity per session
master_df["pages_per_session"] = master_df["total_pages_viewed"] / (master_df["total_sessions"] + 1)

# Total spend to credit limit ratio
master_df["spend_to_credit_ratio"] = master_df["total_spent"] / (master_df["credit_limit"] + 1)

# Impute unmapped geo features for rare customers without session records
master_df["geo_ip_region"] = master_df["geo_ip_region"].fillna("region_unknown")
master_df["region_tier"] = master_df["region_tier"].fillna("TierUnknown")

# Impute geo features using medians derived strictly from training customers
train_master_mask = master_df["customer_id"].isin(train_cust_set)
master_df["average_income"] = master_df["average_income"].fillna(
    master_df.loc[train_master_mask, "average_income"].median()
)
master_df["urban_ratio"] = master_df["urban_ratio"].fillna(
    master_df.loc[train_master_mask, "urban_ratio"].median()
)
master_df["internet_penetration"] = master_df["internet_penetration"].fillna(
    master_df.loc[train_master_mask, "internet_penetration"].median()
)

print("\nUnified Customer Master Feature Dataset:")
print("Shape:", master_df.shape)
print("Unique Customers:", master_df["customer_id"].nunique())
print("Total Missing Values Remaining:", master_df.isnull().sum().sum())
print("\nFirst 3 Rows of Key Engineered Features:")
print(master_df[[
    "customer_id", "customer_tenure_days", "total_sessions", "total_spent",
    "avg_order_value", "cart_to_order_ratio", "spend_to_credit_ratio"
]].head(3))


#=======================
#       Correlation Analysis with Target
#=======================

print("\n" + "=" * 50)
print("Correlation Analysis with Target")
print("=" * 50)

# Merge master features with training targets to evaluate linear associations
train_features = pd.merge(master_df, train_data, on="customer_id", how="inner")

# Select numerical columns excluding customer_id
numeric_cols = train_features.select_dtypes(include=[np.number]).columns.drop("customer_id")
corr_matrix = train_features[numeric_cols].corr()

# Rank features by linear correlation with target (0 = Retained, 1 = Churned)
target_corr = corr_matrix["target"].drop("target").sort_values(ascending=False)

print("\nTop 10 Features Positively Correlated with Churn (Target = 1):")
print(target_corr.head(10).round(4))

print("\nTop 10 Features Negatively Correlated with Churn (Retained, Target = 0):")
print(target_corr.tail(10).round(4))

# Inspect synthetic noise columns correlation
noise_cols = [c for c in target_corr.index if "noise" in c]
if noise_cols:
    print("\nCorrelation of Synthetic Noise Columns with Target (Benchmark):")
    print(target_corr[noise_cols].round(4))


#=======================
#       Multicollinearity Analysis
#=======================

print("\n" + "=" * 50)
print("Multicollinearity Inspection (Highly Correlated Feature Pairs)")
print("=" * 50)

feature_corr = corr_matrix.drop("target", axis=0).drop("target", axis=1)

high_corr_pairs = []
for i in range(len(feature_corr.columns)):
    for j in range(i + 1, len(feature_corr.columns)):
        col1 = feature_corr.columns[i]
        col2 = feature_corr.columns[j]
        val = feature_corr.iloc[i, j]
        if abs(val) > 0.80:
            high_corr_pairs.append((col1, col2, round(val, 4)))

high_corr_df = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Pearson r"])
high_corr_df = high_corr_df.sort_values(by="Pearson r", key=abs, ascending=False).reset_index(drop=True)

print(f"\nTotal Pairs with |r| > 0.80: {len(high_corr_df)}")
print(high_corr_df.head(15))


#=======================
#       Correlation Heatmap Visualization
#=======================

print("\n" + "=" * 50)
print("Correlation Heatmap Visualization")
print("=" * 50)

# Select top correlated features with target for an uncluttered, interpretable heatmap
top_features = target_corr.abs().sort_values(ascending=False).head(14).index.tolist()
heatmap_cols = ["target"] + top_features

plt.figure(figsize=(12, 10))
sns.heatmap(
    train_features[heatmap_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)
plt.title("Correlation Matrix of Top Features with Target")
plt.tight_layout()
plt.show()


#=======================
#       Separate Features and Target (Strict Partitioning)
#=======================

# Merge master feature dataset with train target labels
train_full = pd.merge(master_df, train_data, on="customer_id", how="inner")

drop_cols = [
    "customer_id", "target", "signup_date", "referral_code",
    "noise_customer_0", "noise_customer_1", "noise_customer_2",
    "noise_customer_3", "noise_customer_4",
    "pages_per_session", "signup_year"
]

cat_cols = [
    "country_code", "region", "customer_tier",
    "geo_ip_region", "region_tier", "primary_campaign_type"
]

# Partition strictly using the pre-established customer sets
train_split = train_full[train_full["customer_id"].isin(train_cust_set)].copy()
val_split = train_full[train_full["customer_id"].isin(set(val_cust_ids))].copy()

X_train_raw = train_split.drop(columns=drop_cols)
y_train = train_split["target"]

X_val_raw = val_split.drop(columns=drop_cols)
y_val = val_split["target"]

print("\nX_train Raw Shape:", X_train_raw.shape)
print("X_val Raw Shape:  ", X_val_raw.shape)


#=======================
#       Encode Categories (OneHotEncoder Fit on Train Only)
#=======================

# Fit encoder strictly on X_train to prevent categorical lookahead leakage
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_train_cat = ohe.fit_transform(X_train_raw[cat_cols])
X_val_cat = ohe.transform(X_val_raw[cat_cols])

cat_feature_names = ohe.get_feature_names_out(cat_cols)
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train_raw.index)
X_val_cat_df = pd.DataFrame(X_val_cat, columns=cat_feature_names, index=X_val_raw.index)

X_train = pd.concat([X_train_raw[num_cols], X_train_cat_df], axis=1)
X_val = pd.concat([X_val_raw[num_cols], X_val_cat_df], axis=1)

print("\nFeatures Shape After Encoding (Train):", X_train.shape)
print("Features Shape After Encoding (Val):  ", X_val.shape)


#=======================
#       Scale Features (StandardScaler Fit on Train Only)
#=======================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\nFeatures Scaled Successfully (Fit on Train, Transformed Val).")


#=======================
#       Model Implementation (Logistic Regression)
#=======================

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining Logistic Regression Model...")
model.fit(X_train_scaled, y_train)
print("Model Training Complete.")


#=======================
#       Helper: Optimal Threshold Search
#=======================

def find_optimal_threshold(y_true, y_prob):
    """Searches probability thresholds to maximize F1-score on positive churn class."""
    thresholds = np.linspace(0.10, 0.90, 81)
    best_th = 0.50
    best_f1 = -1.0
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = th
    return round(best_th, 3)


#=======================
#       Model Predictions & Evaluation (Logistic Regression)
#=======================

y_pred_lr = model.predict(X_val_scaled)
y_prob_lr = model.predict_proba(X_val_scaled)[:, 1]

# Default threshold evaluation (p = 0.50)
print("\nLogistic Regression (Default p = 0.50):")
print("Validation Accuracy:", accuracy_score(y_val, y_pred_lr))
print("Classification Report:")
print(classification_report(y_val, y_pred_lr))
print("ROC-AUC Score:", roc_auc_score(y_val, y_prob_lr))

# Optimal threshold evaluation
opt_th_lr = find_optimal_threshold(y_val, y_prob_lr)
y_pred_lr_opt = (y_prob_lr >= opt_th_lr).astype(int)

print(f"\nLogistic Regression (Optimal F1 Threshold p = {opt_th_lr:.3f}):")
print("Validation Accuracy:", accuracy_score(y_val, y_pred_lr_opt))
print("Recall (Churn Caught):", recall_score(y_val, y_pred_lr_opt))
print("F1-Score:             ", f1_score(y_val, y_pred_lr_opt))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_lr_opt))

models_results = [
    {
        "Model": "Logistic Regression (Default)",
        "Algorithm": "Logistic Regression",
        "Threshold": 0.50,
        "Accuracy": accuracy_score(y_val, y_pred_lr),
        "Precision": precision_score(y_val, y_pred_lr, zero_division=0),
        "Recall": recall_score(y_val, y_pred_lr),
        "F1-Score": f1_score(y_val, y_pred_lr),
        "ROC-AUC": roc_auc_score(y_val, y_prob_lr),
        "y_pred": y_pred_lr,
        "y_prob": y_prob_lr
    },
    {
        "Model": f"Logistic Regression (p={opt_th_lr:.2f})",
        "Algorithm": "Logistic Regression",
        "Threshold": opt_th_lr,
        "Accuracy": accuracy_score(y_val, y_pred_lr_opt),
        "Precision": precision_score(y_val, y_pred_lr_opt, zero_division=0),
        "Recall": recall_score(y_val, y_pred_lr_opt),
        "F1-Score": f1_score(y_val, y_pred_lr_opt),
        "ROC-AUC": roc_auc_score(y_val, y_prob_lr),
        "y_pred": y_pred_lr_opt,
        "y_prob": y_prob_lr
    }
]


#=======================
#       Model Implementation (K-Nearest Neighbors)
#=======================

print("\n" + "=" * 50)
print("Model Implementation: K-Nearest Neighbors (KNN)")
print("=" * 50)

knn_model = KNeighborsClassifier(n_neighbors=5)

print("\nTraining K-Nearest Neighbors Model...")
knn_model.fit(X_train_scaled, y_train)
print("KNN Training Complete.")

y_prob_knn = knn_model.predict_proba(X_val_scaled)[:, 1]
y_pred_knn = (y_prob_knn >= 0.5).astype(int)

print("\nKNN Validation Accuracy:", accuracy_score(y_val, y_pred_knn))
print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_knn))
print("\nKNN Classification Report:")
print(classification_report(y_val, y_pred_knn))
print("KNN ROC-AUC Score:", roc_auc_score(y_val, y_prob_knn))

models_results.append({
    "Model": "K-Nearest Neighbors",
    "Algorithm": "K-Nearest Neighbors",
    "Threshold": 0.50,
    "Accuracy": accuracy_score(y_val, y_pred_knn),
    "Precision": precision_score(y_val, y_pred_knn, zero_division=0),
    "Recall": recall_score(y_val, y_pred_knn),
    "F1-Score": f1_score(y_val, y_pred_knn),
    "ROC-AUC": roc_auc_score(y_val, y_prob_knn),
    "y_pred": y_pred_knn,
    "y_prob": y_prob_knn
})


#=======================
#       Model Implementation (Naive Bayes)
#=======================

print("\n" + "=" * 50)
print("Model Implementation: Naive Bayes (GaussianNB)")
print("=" * 50)

nb_model = GaussianNB()

print("\nTraining Naive Bayes Model...")
nb_model.fit(X_train_scaled, y_train)
print("Naive Bayes Training Complete.")

y_pred_nb = nb_model.predict(X_val_scaled)
y_prob_nb = nb_model.predict_proba(X_val_scaled)[:, 1]

print("\nNaive Bayes Validation Accuracy:", accuracy_score(y_val, y_pred_nb))
print("\nNaive Bayes Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_nb))
print("\nNaive Bayes Classification Report:")
print(classification_report(y_val, y_pred_nb))
print("Naive Bayes ROC-AUC Score:", roc_auc_score(y_val, y_prob_nb))

models_results.append({
    "Model": "Naive Bayes",
    "Algorithm": "Naive Bayes",
    "Threshold": 0.50,
    "Accuracy": accuracy_score(y_val, y_pred_nb),
    "Precision": precision_score(y_val, y_pred_nb, zero_division=0),
    "Recall": recall_score(y_val, y_pred_nb),
    "F1-Score": f1_score(y_val, y_pred_nb),
    "ROC-AUC": roc_auc_score(y_val, y_prob_nb),
    "y_pred": y_pred_nb,
    "y_prob": y_prob_nb
})


#=======================
#       Model Implementation (Decision Tree)
#=======================

print("\n" + "=" * 50)
print("Model Implementation: Decision Tree")
print("=" * 50)

dt_model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining Decision Tree Model...")
dt_model.fit(X_train_scaled, y_train)
print("Decision Tree Training Complete.")

y_pred_dt = dt_model.predict(X_val_scaled)
y_prob_dt = dt_model.predict_proba(X_val_scaled)[:, 1]

# Default threshold evaluation (p = 0.50)
print("\nDecision Tree (Default p = 0.50):")
print("Validation Accuracy:", accuracy_score(y_val, y_pred_dt))
print("Classification Report:")
print(classification_report(y_val, y_pred_dt))
print("ROC-AUC Score:", roc_auc_score(y_val, y_prob_dt))

# Optimal threshold evaluation
opt_th_dt = find_optimal_threshold(y_val, y_prob_dt)
y_pred_dt_opt = (y_prob_dt >= opt_th_dt).astype(int)

print(f"\nDecision Tree (Optimal F1 Threshold p = {opt_th_dt:.3f}):")
print("Validation Accuracy:", accuracy_score(y_val, y_pred_dt_opt))
print("Recall (Churn Caught):", recall_score(y_val, y_pred_dt_opt))
print("F1-Score:             ", f1_score(y_val, y_pred_dt_opt))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_dt_opt))

models_results.extend([
    {
        "Model": "Decision Tree (Default)",
        "Algorithm": "Decision Tree",
        "Threshold": 0.50,
        "Accuracy": accuracy_score(y_val, y_pred_dt),
        "Precision": precision_score(y_val, y_pred_dt, zero_division=0),
        "Recall": recall_score(y_val, y_pred_dt),
        "F1-Score": f1_score(y_val, y_pred_dt),
        "ROC-AUC": roc_auc_score(y_val, y_prob_dt),
        "y_pred": y_pred_dt,
        "y_prob": y_prob_dt
    },
    {
        "Model": f"Decision Tree (p={opt_th_dt:.2f})",
        "Algorithm": "Decision Tree",
        "Threshold": opt_th_dt,
        "Accuracy": accuracy_score(y_val, y_pred_dt_opt),
        "Precision": precision_score(y_val, y_pred_dt_opt, zero_division=0),
        "Recall": recall_score(y_val, y_pred_dt_opt),
        "F1-Score": f1_score(y_val, y_pred_dt_opt),
        "ROC-AUC": roc_auc_score(y_val, y_prob_dt),
        "y_pred": y_pred_dt_opt,
        "y_prob": y_prob_dt
    }
])


#=======================
#       Model Performance Comparison
#=======================

print("\n" + "=" * 50)
print("Model Evaluation & Metrics Comparison Summary")
print("=" * 50)

comparison_df = pd.DataFrame([{
    "Model": r["Model"],
    "Threshold": f"{r['Threshold']:.2f}",
    "Accuracy": round(r["Accuracy"], 4),
    "Precision": round(r["Precision"], 4),
    "Recall": round(r["Recall"], 4),
    "F1-Score": round(r["F1-Score"], 4),
    "ROC-AUC": round(r["ROC-AUC"], 4)
} for r in models_results])

print("\nSummary Comparison Table (Ranked by F1-Score):")
print(comparison_df.sort_values(by="F1-Score", ascending=False).to_string(index=False))

best_f1_entry = comparison_df.sort_values(by="F1-Score", ascending=False).iloc[0]
best_auc_entry = comparison_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]
print(f"\nBest Model by F1-Score: {best_f1_entry['Model']} (F1 = {best_f1_entry['F1-Score']:.4f}, Recall = {best_f1_entry['Recall']:.4f})")
print(f"Best Model by ROC-AUC:  {best_auc_entry['Model']} (AUC = {best_auc_entry['ROC-AUC']:.4f})")


#=======================
#       Comparative Visualizations
#=======================

# 1. Bar Chart: Performance Metrics Across Models
metrics_melted = comparison_df.melt(
    id_vars=["Model"],
    value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(14, 6))
sns.barplot(x="Metric", y="Score", hue="Model", data=metrics_melted, palette="Set2")
plt.title("Model Performance Metrics Comparison Across Decision Thresholds", fontsize=14, pad=12)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.xlabel("Evaluation Metric")
plt.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# 2. ROC Curves Comparison (Unique Algorithms)
plt.figure(figsize=(9, 7))
seen_algos = set()
for r in models_results:
    algo = r["Algorithm"]
    if algo not in seen_algos:
        seen_algos.add(algo)
        fpr, tpr, _ = roc_curve(y_val, r["y_prob"])
        plt.plot(fpr, tpr, label=f"{algo} (AUC = {r['ROC-AUC']:.4f})", linewidth=2)

plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)", alpha=0.7)
plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, pad=12)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# 3. Confusion Matrices 2x2 Subplots (Threshold-Optimized Operational Grid)
plot_models = [
    [r for r in models_results if r["Model"].startswith("Decision Tree (p=")][0],
    [r for r in models_results if r["Model"].startswith("Logistic Regression (p=")][0],
    [r for r in models_results if r["Model"] == "K-Nearest Neighbors"][0],
    [r for r in models_results if r["Model"] == "Naive Bayes"][0]
]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()

for idx, r in enumerate(plot_models):
    cm = confusion_matrix(y_val, r["y_pred"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[idx],
        annot_kws={"size": 13}
    )
    axes[idx].set_title(f"{r['Model']}\nConfusion Matrix", fontsize=11)
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("Actual Label")
    axes[idx].set_xticklabels(["Retained (0)", "Churned (1)"])
    axes[idx].set_yticklabels(["Retained (0)", "Churned (1)"])

plt.tight_layout()
plt.show()

#=======================
#       Check Model Variables
#=======================

print("Available model variables:")

for name, value in globals().copy().items():
    if "model" in name.lower() or "tree" in name.lower():
        if hasattr(value, "predict"):
            print(name, "->", type(value).__name__)
            
#=======================
#       Check Model Readiness
#=======================


models_to_check = {
    "Logistic Regression": model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Decision Tree": dt_model
}

for name, clf in models_to_check.items():
    try:
        clf.predict(X_train_scaled[:1])
        print(name, "-> Fitted and ready")
    except Exception as e:
        print(name, "-> Not ready:", str(e))

print("Preprocessor available:", "preprocessor" in globals())
print("Raw training data available:", "X_train" in globals())
print("Processed training data available:", "X_train_processed" in globals())

#=======================
#       Check Preprocessing Variables
#=======================

for name in globals():
    if any(word in name.lower() for word in [
        "preprocess", "transform", "scaled", "encoded", "imputer"
    ]):
        print(name)
        
#===============================================================================================

import joblib

#=======================
#       Save Model Artifacts
#=======================

# Create folder for saved model files
os.makedirs("models", exist_ok=True)

# Save model, encoder, scaler, feature columns, and threshold
joblib.dump(
    {
        "model": dt_model,
        "encoder": ohe,
        "scaler": scaler,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "feature_columns": X_train.columns.tolist(),
        "threshold": opt_th_dt
    },
    "models/retention_model.pkl"
)

print("\nModel artifacts saved successfully!")
print("File: models/retention_model.pkl")
print("Decision Tree threshold:", opt_th_dt)

#=======================
#       Verify Saved Model
#=======================

saved_artifacts = joblib.load("models/retention_model.pkl")

print("\nSaved Model Verification:")
print("Model:", type(saved_artifacts["model"]).__name__)
print("Encoder:", type(saved_artifacts["encoder"]).__name__)
print("Scaler:", type(saved_artifacts["scaler"]).__name__)
print("Feature count:", len(saved_artifacts["feature_columns"]))
print("Threshold:", saved_artifacts["threshold"])

#=======================
#       Test Model Prediction
#=======================

# Select one validation customer
sample_raw = X_val_raw.iloc[[0]].copy()

# Apply the saved encoder
sample_cat = saved_artifacts["encoder"].transform(
    sample_raw[saved_artifacts["categorical_columns"]]
)

sample_cat_df = pd.DataFrame(
    sample_cat,
    columns=saved_artifacts["encoder"].get_feature_names_out(
        saved_artifacts["categorical_columns"]
    ),
    index=sample_raw.index
)

# Rebuild features in the original training order
sample_numeric = sample_raw[saved_artifacts["numeric_columns"]]

sample_features = pd.concat([sample_numeric, sample_cat_df], axis=1)
sample_features = sample_features[saved_artifacts["feature_columns"]]

# Apply saved scaler
sample_scaled = saved_artifacts["scaler"].transform(sample_features)

# Predict probability and class
probability = saved_artifacts["model"].predict_proba(sample_scaled)[0, 1]
prediction = int(probability >= saved_artifacts["threshold"])

print("\nTest Prediction:")
print("Predicted probability of class 1:", round(probability, 4))
print("Predicted class:", prediction)
print("Actual class:", int(y_val.iloc[0]))

#=======================
#       Export Test Input
#=======================

sample_input = X_val_raw.iloc[0][num_cols + cat_cols]

print("\nSample API Input:")
print(sample_input.to_json())