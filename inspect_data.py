#=======================
#       Import Libraries
#=======================
import numpy as np
import pandas as pd
import os


#=======================
#       Read File Names
#=======================

path = "data/raw"

files = os.listdir(path)

print(files)

#=======================
#       Inspect CSV Files
#=======================

for file in files:
    if file.endswith(".csv"):
        data = pd.read_csv(path + "/" + file)

        print("\nFile Name:", file)
        print("Columns:", data.columns.tolist())
        print("Shape:", data.shape)
        print("Info: \n",data.info())
        print("Describe: \n",data.describe())
        print(data.head(2))

#=======================
#       Check Missing Values
#=======================

for file in files:
    if file.endswith(".csv"):
        data = pd.read_csv(path + "/" + file)

        print("\nFile Name:", file)
        print(data.isnull().sum())

#=======================
#       Check Duplicates
#=======================

for file in files:
    if file.endswith(".csv"):
        data = pd.read_csv(path + "/" + file)

        print("\nFile Name:", file)
        print("Duplicate Rows:", data.duplicated().sum())


#=======================
#       Target Analysis
#=======================

data = pd.read_csv(path + "/train.csv")

print("\nTarget Distribution:")
print(data["target"].value_counts())

print("\nTarget Percentage:")
print(data["target"].value_counts(normalize=True) * 100)

#=======================
#       Target Visualization
#=======================

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(path + "/train.csv")

sns.countplot(x="target", data=data)
plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()


#=======================
#       Numerical Feature Analysis
#=======================

data = pd.read_csv(path + "/customers.csv")

print("\nNumerical Feature Summary:")
print(data.describe())

#=======================
#       Numerical Feature Visualization
#=======================

data[["loyalty_score", "email_open_rate",
      "discount_usage_rate", "avg_review_score"]].hist(
          figsize=(10, 8), bins=30)

plt.tight_layout()
plt.show()


#=======================
#       Categorical Feature Analysis
#=======================

data = pd.read_csv(path + "/customers.csv")

print("\nCountry Code Distribution:")
print(data["country_code"].value_counts().head(10))

print("\nRegion Distribution:")
print(data["region"].value_counts().head(10))

print("\nCustomer Tier Distribution:")
print(data["customer_tier"].value_counts())

print("\nReferral Code Distribution:")
print(data["referral_code"].value_counts().head(10))


#=======================
#       Session Data Analysis
#=======================

data = pd.read_csv(path + "/sessions.csv")

print("\nSession Data Summary:")
print(data[["session_duration", "pages_viewed",
            "cart_additions"]].describe())

print("\nTraffic Source Distribution:")
print(data["traffic_source"].value_counts())

print("\nDevice Type Distribution:")
print(data["device_type"].value_counts())

print("\nBounce Flag Distribution:")
print(data["bounce_flag"].value_counts())


#=======================
#       Transaction Data Analysis
#=======================

data = pd.read_csv(path + "/transactions.csv")

print("\nTransaction Summary:")
print(data[["order_value", "items_count"]].describe())

print("\nPayment Method Distribution:")
print(data["payment_method"].value_counts())

print("\nDiscount Applied Distribution:")
print(data["discount_applied"].value_counts())

print("\nShipping Speed Distribution:")
print(data["shipping_speed"].value_counts())

print("\nHigh Value Transaction Distribution:")
print(data["high_value_flag"].value_counts())


#=======================
#       Geographical Data Analysis
#=======================

data = pd.read_csv(path + "/geo_data.csv")

print("\nGeographical Data Summary:")
print(data.describe())

print("\nRegion Tier Distribution:")
print(data["region_tier"].value_counts())

#=======================
#       Geographical Visualization
#=======================

data[["average_income", "urban_ratio",
      "internet_penetration"]].hist(
          figsize=(10, 6), bins=20)

plt.tight_layout()
plt.show()


#=======================
#       Marketing Campaign Analysis
#=======================

data = pd.read_csv(path + "/marketing_campaigns.csv")

print("\nCampaign Type Distribution:")
print(data["campaign_type"].value_counts())

print("\nCampaign Budget Summary:")
print(data["campaign_budget"].describe())

print("\nTarget Region Distribution:")
print(data["region_target"].value_counts().head(10))

print("\nCampaign Dates:")
print(data[["start_date", "end_date"]].head())



#=======================
#       Marketing Campaign Analysis
#=======================

data = pd.read_csv(path + "/marketing_campaigns.csv")

print("\nCampaign Type Distribution:")
print(data["campaign_type"].value_counts())

print("\nCampaign Budget Summary:")
print(data["campaign_budget"].describe())

print("\nTarget Region Distribution:")
print(data["region_target"].value_counts().head(10))

print("\nCampaign Dates:")
print(data[["start_date", "end_date"]].head())


#=======================
#       Customer Target Analysis
#=======================

customers = pd.read_csv(path + "/customers.csv")
train = pd.read_csv(path + "/train.csv")

data = pd.merge(customers, train, on="customer_id")

print("\nAverage Features by Target:")
print(data.groupby("target")[
    ["loyalty_score", "discount_usage_rate",
     "avg_review_score"]
].mean())

#=======================
#       Target Comparison Plot
#=======================

sns.barplot(x="target", y="loyalty_score", data=data)
plt.title("Loyalty Score by Target")
plt.show()


#=======================
#       Load Modeling Data
#=======================

customers = pd.read_csv(path + "/customers.csv")
train = pd.read_csv(path + "/train.csv")

data = pd.merge(customers, train, on="customer_id")

print("\nModeling Dataset Shape:")
print(data.shape)

print("\nTarget Distribution:")
print(data["target"].value_counts())

print("\nFirst Five Rows:")
print(data.head())


#=======================
#       Check Modeling Data Missing Values
#=======================

print("\nMissing Values:")
print(data.isnull().sum()[data.isnull().sum() > 0])

print("\nTotal Missing Values:")
print(data.isnull().sum().sum())


#=======================
#       Handle Missing Values
#=======================

median_value = data["email_open_rate"].median()

data["email_open_rate"] = data["email_open_rate"].fillna(median_value)

print("\nMissing Values After Treatment:")
print(data.isnull().sum().sum())


#=======================
#       Separate Features and Target
#=======================

X = data.drop(["customer_id", "target"], axis=1)
y = data["target"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())


#=======================
#       Identify Categorical Columns
#=======================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

print("\nCategorical Columns:")
print(categorical_columns.tolist())

print("\nNumber of Categorical Columns:")
print(len(categorical_columns))


#=======================
#       Inspect Categorical Values
#=======================

for column in categorical_columns:
    print("\nColumn:", column)
    print("Unique Values:", X[column].nunique())
    print(X[column].head(5).tolist())


#=======================
#       Convert Data Types
#=======================

X["age"] = pd.to_numeric(X["age"], errors="coerce")

X["signup_date"] = pd.to_datetime(
    X["signup_date"], errors="coerce"
)

X["signup_year"] = X["signup_date"].dt.year
X["signup_month"] = X["signup_date"].dt.month

X = X.drop("signup_date", axis=1)

X["credit_limit"] = (
    X["credit_limit"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

X["credit_limit"] = pd.to_numeric(
    X["credit_limit"], errors="coerce"
)

print("\nUpdated Data Types:")
print(X[["age", "signup_year", "signup_month",
         "credit_limit"]].dtypes)

print("\nMissing Values:")
print(X[["age", "signup_year", "signup_month",
         "credit_limit"]].isnull().sum())


#=======================
#       Handle Missing Age
#=======================

median_age = X["age"].median()

X["age"] = X["age"].fillna(median_age)

print("\nMissing Age Values After Treatment:")
print(X["age"].isnull().sum())


#=======================
#       Clean and Encode Categories
#=======================

X["country_code"] = X["country_code"].replace({
    "usa": "USA",
    "U.S.": "USA"
})

X = X.drop("referral_code", axis=1)

X = pd.get_dummies(
    X,
    columns=["country_code", "region", "customer_tier"],
    dtype=int
)

print("\nUpdated Dataset Shape:")
print(X.shape)

print("\nRemaining Object Columns:")
print(X.select_dtypes(include=["object"]).columns.tolist())


#=======================
#       Final Missing Value Check
#=======================

print("\nMissing Values in X:")
print(X.isnull().sum().sum())

print("\nMissing Values in y:")
print(y.isnull().sum())


#=======================
#       Check Date Ranges
#=======================

customers = pd.read_csv(path + "/customers.csv")
sessions = pd.read_csv(path + "/sessions.csv")
transactions = pd.read_csv(path + "/transactions.csv")

print("\nCustomer Signup Date Range:")
print(customers["signup_date"].min())
print(customers["signup_date"].max())

print("\nSession Timestamp Range:")
print(sessions["session_timestamp"].min())
print(sessions["session_timestamp"].max())

print("\nTransaction Timestamp Range:")
print(transactions["transaction_timestamp"].min())
print(transactions["transaction_timestamp"].max())



#=======================
#       Check Customer ID Relationships
#=======================

customers = pd.read_csv(path + "/customers.csv")
sessions = pd.read_csv(path + "/sessions.csv")
transactions = pd.read_csv(path + "/transactions.csv")
train = pd.read_csv(path + "/train.csv")

print("\nCustomer ID Counts:")
print("Customers:", customers["customer_id"].nunique())
print("Training Customers:", train["customer_id"].nunique())

print("\nSession Customer ID Check:")
print(sessions["customer_id"].isin(
    customers["customer_id"]
).all())

print("\nTransaction Customer ID Check:")
print(transactions["customer_id"].isin(
    customers["customer_id"]
).all())

print("\nTraining Customer ID Check:")
print(train["customer_id"].isin(
    customers["customer_id"]
).all())


#=======================
#       Check Customer Activity Coverage
#=======================

print("\nCustomer Session Coverage:")
print("Customers with sessions:",
      sessions["customer_id"].nunique())

print("\nCustomer Transaction Coverage:")
print("Customers with transactions:",
      transactions["customer_id"].nunique())

print("\nTraining Customers with Sessions:")
print(train["customer_id"].isin(
    sessions["customer_id"]
).sum())

print("\nTraining Customers with Transactions:")
print(train["customer_id"].isin(
    transactions["customer_id"]
).sum())


#=======================
#       Check Region and Campaign Relationships
#=======================

geo = pd.read_csv(path + "/geo_data.csv")
campaigns = pd.read_csv(path + "/marketing_campaigns.csv")
sessions = pd.read_csv(path + "/sessions.csv")

print("\nUnique Session Regions:",
      sessions["geo_ip_region"].nunique())

print("Unique Geo Data Regions:",
      geo["geo_ip_region"].nunique())

print("\nAll Session Regions Match Geo Data:")
print(sessions["geo_ip_region"].isin(
    geo["geo_ip_region"]
).all())

print("\nUnique Session Campaign IDs:",
      sessions["campaign_id"].nunique())

print("Unique Campaign IDs:")
print(campaigns["campaign_id"].nunique())

print("\nAll Session Campaign IDs Match Campaign Data:")
print(sessions["campaign_id"].isin(
    campaigns["campaign_id"]
).all())


#=======================
#       Check Campaign Region Matching
#=======================

campaigns = pd.read_csv(path + "/marketing_campaigns.csv")
sessions = pd.read_csv(path + "/sessions.csv")

print("\nCampaign Target Regions:")
print(campaigns["region_target"].head(10).tolist())

print("\nSession Geographic Regions:")
print(sessions["geo_ip_region"].head(10).tolist())

print("\nUnique Campaign Target Regions:")
print(campaigns["region_target"].nunique())

print("\nUnique Session Geographic Regions:")
print(sessions["geo_ip_region"].nunique())

print("\nAll Campaign Regions Match Session Regions:")
print(campaigns["region_target"].isin(
    sessions["geo_ip_region"]
).all())


#=======================
#       Check Campaign and Session Region Consistency
#=======================

campaigns = pd.read_csv(path + "/marketing_campaigns.csv")
sessions = pd.read_csv(path + "/sessions.csv")

data = pd.merge(
    sessions[["campaign_id", "geo_ip_region"]],
    campaigns[["campaign_id", "region_target"]],
    on="campaign_id",
    how="left"
)

print("\nTotal Sessions:", len(data))

print("\nCampaign Region Matches Session Region:")
print((data["geo_ip_region"] == data["region_target"]).sum())

print("\nCampaign Region Does Not Match Session Region:")
print((data["geo_ip_region"] != data["region_target"]).sum())


#=======================
#       Inspect Campaign Data
#=======================

campaigns = pd.read_csv(path + "/marketing_campaigns.csv")

print("\nCampaign Data Types:")
print(campaigns.dtypes)

print("\nCampaign Types:")
print(campaigns["campaign_type"].value_counts())

print("\nCampaign Budget Values:")
print(campaigns["campaign_budget"].head(10).tolist())

print("\nCampaign Date Range:")
print(campaigns[["start_date", "end_date"]].describe())


#=======================
#       Check Campaign Date Coverage
#=======================

campaigns = pd.read_csv(path + "/marketing_campaigns.csv")
sessions = pd.read_csv(path + "/sessions.csv")

campaigns["start_date"] = pd.to_datetime(campaigns["start_date"])
campaigns["end_date"] = pd.to_datetime(campaigns["end_date"])
sessions["session_timestamp"] = pd.to_datetime(
    sessions["session_timestamp"]
)

print("\nCampaign Start Date Range:")
print(campaigns["start_date"].min())
print(campaigns["start_date"].max())

print("\nCampaign End Date Range:")
print(campaigns["end_date"].min())
print(campaigns["end_date"].max())

print("\nSession Date Range:")
print(sessions["session_timestamp"].min())
print(sessions["session_timestamp"].max())


#=======================
#       Check Active Campaign Sessions
#=======================

campaigns = pd.read_csv(path + "/marketing_campaigns.csv")
sessions = pd.read_csv(path + "/sessions.csv")

campaigns["start_date"] = pd.to_datetime(campaigns["start_date"])
campaigns["end_date"] = pd.to_datetime(campaigns["end_date"])
sessions["session_timestamp"] = pd.to_datetime(
    sessions["session_timestamp"]
)

data = pd.merge(
    sessions[["campaign_id", "session_timestamp"]],
    campaigns[["campaign_id", "start_date", "end_date"]],
    on="campaign_id",
    how="left"
)

active_sessions = (
    (data["session_timestamp"] >= data["start_date"]) &
    (data["session_timestamp"] <= data["end_date"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
)

print("\nSessions During Campaign Dates:")
print(active_sessions.sum())

print("\nSessions Outside Campaign Dates:")
print((~active_sessions).sum())

print("\nPercentage During Campaign Dates:")
print(active_sessions.mean() * 100)


#=======================
#       Transaction Date Analysis
#=======================

transactions = pd.read_csv(path + "/transactions.csv")

transactions["transaction_timestamp"] = pd.to_datetime(
    transactions["transaction_timestamp"]
)

print("\nTransaction Date Range:")
print(transactions["transaction_timestamp"].min())
print(transactions["transaction_timestamp"].max())

print("\nTransactions Per Year:")
print(
    transactions["transaction_timestamp"]
    .dt.year.value_counts()
    .sort_index()
)

print("\nTransactions Per Month:")
print(
    transactions["transaction_timestamp"]
    .dt.to_period("M")
    .value_counts()
    .sort_index()
)

#=======================
#       Transaction Frequency Analysis
#=======================

transactions = pd.read_csv(path + "/transactions.csv")

transaction_count = transactions.groupby(
    "customer_id"
).size()

print("\nTransaction Count Summary:")
print(transaction_count.describe())

print("\nCustomers with One Transaction:")
print((transaction_count == 1).sum())

print("\nCustomers with More Than One Transaction:")
print((transaction_count > 1).sum())

print("\nCustomers with No Transactions:")
print(
    200000 - transactions["customer_id"].nunique()
)

#=======================
#       Transaction Value Analysis
#=======================

transactions = pd.read_csv(path + "/transactions.csv")

transactions["order_value"] = (
    transactions["order_value"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

transactions["order_value"] = pd.to_numeric(
    transactions["order_value"], errors="coerce"
)

print("\nOrder Value Summary:")
print(transactions["order_value"].describe())

print("\nItems Count Summary:")
print(transactions["items_count"].describe())


#=======================
#       Transaction Outlier Detection
#=======================

transactions = pd.read_csv(path + "/transactions.csv")

transactions["order_value"] = (
    transactions["order_value"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

transactions["order_value"] = pd.to_numeric(
    transactions["order_value"], errors="coerce"
)

for column in ["order_value", "items_count"]:
    Q1 = transactions[column].quantile(0.25)
    Q3 = transactions[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = transactions[
        (transactions[column] < lower_limit) |
        (transactions[column] > upper_limit)
    ]

    print("\nColumn:", column)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Outlier Count:", len(outliers))


#=======================
#       Session Behavior Analysis
#=======================

sessions = pd.read_csv(path + "/sessions.csv")

print("\nSession Duration Summary:")
print(sessions["session_duration"].describe())

print("\nPages Viewed Summary:")
print(sessions["pages_viewed"].describe())

print("\nCart Additions Summary:")
print(sessions["cart_additions"].describe())

print("\nBounce Flag Distribution:")
print(sessions["bounce_flag"].value_counts())


#=======================
#       Session Outlier Detection
#=======================

sessions = pd.read_csv(path + "/sessions.csv")

for column in [
    "session_duration",
    "pages_viewed",
    "cart_additions"
]:
    Q1 = sessions[column].quantile(0.25)
    Q3 = sessions[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = sessions[
        (sessions[column] < lower_limit) |
        (sessions[column] > upper_limit)
    ]

    print("\nColumn:", column)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Outlier Count:", len(outliers))


#=======================
#       Session Category Analysis
#=======================

sessions = pd.read_csv(path + "/sessions.csv")

for column in [
    "traffic_source",
    "device_type",
    "bounce_flag"
]:
    print("\nColumn:", column)
    print(sessions[column].value_counts(dropna=False))


#=======================
#       Customer Activity Analysis
#=======================

sessions = pd.read_csv(path + "/sessions.csv")
transactions = pd.read_csv(path + "/transactions.csv")

session_count = sessions.groupby("customer_id").size()
transaction_count = transactions.groupby("customer_id").size()

print("\nSessions Per Customer:")
print(session_count.describe())

print("\nTransactions Per Customer:")
print(transaction_count.describe())

print("\nCustomers with More Than 10 Sessions:")
print((session_count > 10).sum())

print("\nCustomers with More Than 5 Transactions:")
print((transaction_count > 5).sum())


#=======================
#       Check Customer and Session Regions
#=======================

customers = pd.read_csv(path + "/customers.csv")
sessions = pd.read_csv(path + "/sessions.csv")

print("\nCustomer Regions:")
print(customers["region"].nunique())

print("\nSession Regions:")
print(sessions["geo_ip_region"].nunique())

print("\nCustomer Region Examples:")
print(customers["region"].head(10).tolist())

print("\nSession Region Examples:")
print(sessions["geo_ip_region"].head(10).tolist())

print("\nCustomer Regions Matching Geo Data:")
geo = pd.read_csv(path + "/geo_data.csv")

print(customers["region"].isin(
    geo["geo_ip_region"]
).all())

#=======================
#       Check Activity Before Signup
#=======================

customers = pd.read_csv(path + "/customers.csv")
sessions = pd.read_csv(path + "/sessions.csv")
transactions = pd.read_csv(path + "/transactions.csv")

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)
sessions["session_timestamp"] = pd.to_datetime(
    sessions["session_timestamp"]
)
transactions["transaction_timestamp"] = pd.to_datetime(
    transactions["transaction_timestamp"]
)

data = pd.merge(
    sessions[["customer_id", "session_timestamp"]],
    customers[["customer_id", "signup_date"]],
    on="customer_id",
    how="left"
)

print("\nSessions Before Signup:")
print((data["session_timestamp"] < data["signup_date"]).sum())

data = pd.merge(
    transactions[["customer_id", "transaction_timestamp"]],
    customers[["customer_id", "signup_date"]],
    on="customer_id",
    how="left"
)

print("\nTransactions Before Signup:")
print((data["transaction_timestamp"] < data["signup_date"]).sum())


#=======================
#       Analyze Pre-Signup Activity
#=======================

customers = pd.read_csv(path + "/customers.csv")
sessions = pd.read_csv(path + "/sessions.csv")

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)
sessions["session_timestamp"] = pd.to_datetime(
    sessions["session_timestamp"]
)

data = pd.merge(
    sessions[["customer_id", "session_timestamp"]],
    customers[["customer_id", "signup_date"]],
    on="customer_id",
    how="left"
)

data["before_signup"] = (
    data["session_timestamp"] < data["signup_date"]
)

customer_check = data.groupby("customer_id")[
    "before_signup"
].any()

print("\nCustomers with Any Pre-Signup Sessions:")
print(customer_check.sum())

print("\nCustomers with No Pre-Signup Sessions:")
print((~customer_check).sum())


#=======================
#       Measure Pre-Signup Date Gap
#=======================

customers = pd.read_csv(path + "/customers.csv")
sessions = pd.read_csv(path + "/sessions.csv")

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)
sessions["session_timestamp"] = pd.to_datetime(
    sessions["session_timestamp"]
)

data = pd.merge(
    sessions[["customer_id", "session_timestamp"]],
    customers[["customer_id", "signup_date"]],
    on="customer_id",
    how="left"
)

data = data[
    data["session_timestamp"] < data["signup_date"]
].copy()

data["days_before_signup"] = (
    data["signup_date"] - data["session_timestamp"]
).dt.days

print("\nPre-Signup Gap Summary (Days):")
print(data["days_before_signup"].describe())


#=======================
#       Target Relationship Analysis
#=======================

customers = pd.read_csv(path + "/customers.csv")
train = pd.read_csv(path + "/train.csv")

data = pd.merge(customers, train, on="customer_id")

print("\nAverage Features by Target:")
print(data.groupby("target")[
    ["loyalty_score",
     "discount_usage_rate",
     "avg_review_score",
     "email_open_rate"]
].mean())

print("\nMedian Features by Target:")
print(data.groupby("target")[
    ["loyalty_score",
     "discount_usage_rate",
     "avg_review_score",
     "email_open_rate"]
].median())


#=======================
#       Correlation Analysis
#=======================

customers = pd.read_csv(path + "/customers.csv")
train = pd.read_csv(path + "/train.csv")

data = pd.merge(customers, train, on="customer_id")

numeric_data = data.select_dtypes(include=["number"])

correlation = numeric_data.corr()

print("\nCorrelation with Target:")
print(
    correlation["target"]
    .drop("target")
    .sort_values(ascending=False)
)

#=======================
#       Correlation Heatmap
#=======================

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation,
    anote=True,
    cmap="coolwarm"
)

plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.show()

#=======================
#       Customer Data Integration
#=======================

customers = pd.read_csv(path + "/customers.csv")
train = pd.read_csv(path + "/train.csv")

customer_data = pd.merge(
    customers,
    train,
    on="customer_id",
    how="inner"
)

print("Customer Data Shape:", customer_data.shape)
print("Unique Customer IDs:", customer_data["customer_id"].nunique())
print("Missing Values:", customer_data.isnull().sum().sum())
print("Duplicate Customer IDs:", customer_data["customer_id"].duplicated().sum())

#=======================
#       Session Data Aggregation
#=======================

sessions = pd.read_csv(path + "/sessions.csv")

session_data = sessions.groupby("customer_id").agg(
    total_sessions=("session_id", "count"),
    avg_session_duration=("session_duration", "mean"),
    total_pages_viewed=("pages_viewed", "sum"),
    avg_pages_viewed=("pages_viewed", "mean"),
    total_cart_additions=("cart_additions", "sum"),
    avg_cart_additions=("cart_additions", "mean")
).reset_index()

print("Session Data Shape:", session_data.shape)
print("Unique Customer IDs:", session_data["customer_id"].nunique())
print("Missing Values:", session_data.isnull().sum().sum())
print(session_data.head())

#=======================
#       Transaction Data Aggregation
#=======================

transactions = pd.read_csv(path + "/transactions.csv")

# Convert order_value from text to numeric
transactions["order_value"] = (
    transactions["order_value"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

transaction_data = transactions.groupby("customer_id").agg(
    total_transactions=("transaction_id", "count"),
    total_spent=("order_value", "sum"),
    avg_order_value=("order_value", "mean"),
    max_order_value=("order_value", "max"),
    total_items=("items_count", "sum"),
    avg_items_per_order=("items_count", "mean")
).reset_index()

print("Transaction Data Shape:", transaction_data.shape)
print("Unique Customer IDs:", transaction_data["customer_id"].nunique())
print("Missing Values:", transaction_data.isnull().sum().sum())
print(transaction_data.head())

#=======================
#       Customer Geography Aggregation
#=======================

# Find the most frequent region for each customer
customer_region = (
    sessions.groupby(["customer_id", "geo_ip_region"])
    .size()
    .reset_index(name="region_visits")
)

customer_region = customer_region.sort_values(
    ["customer_id", "region_visits"],
    ascending=[True, False]
)

customer_region = customer_region.drop_duplicates(
    subset="customer_id"
)

customer_region = customer_region[
    ["customer_id", "geo_ip_region"]
]

print("Customer Region Shape:", customer_region.shape)
print("Unique Customer IDs:", customer_region["customer_id"].nunique())
print(customer_region.head())

#=======================
#       Traffic and Device Aggregation
#=======================

# Standardize text categories
sessions["traffic_source"] = sessions["traffic_source"].str.lower()
sessions["device_type"] = sessions["device_type"].str.lower()

traffic_data = sessions.groupby("customer_id").agg(
    unique_traffic_sources=("traffic_source", "nunique"),
    unique_device_types=("device_type", "nunique"),
    avg_bounce_flag=("bounce_flag", lambda x: x.astype(str).str.lower().isin(["1", "true", "yes"]).mean())
).reset_index()

print("Traffic Data Shape:", traffic_data.shape)
print("Unique Customer IDs:", traffic_data["customer_id"].nunique())
print("Missing Values:", traffic_data.isnull().sum().sum())
print(traffic_data.head())

#=======================
#       Geographic Data Integration
#=======================

geo_data = pd.read_csv(path + "/geo_data.csv")

customer_region_data = pd.merge(
    customer_region,
    geo_data,
    on="geo_ip_region",
    how="left"
)

print("Customer Geography Shape:", customer_region_data.shape)
print("Missing Values:", customer_region_data.isnull().sum().sum())
print("Unique Customer IDs:", customer_region_data["customer_id"].nunique())
print(customer_region_data.head())

#=======================
#       Campaign Data Aggregation
#=======================

campaigns = pd.read_csv(path + "/marketing_campaigns.csv")

# Convert campaign budget to numeric
campaigns["campaign_budget"] = (
    campaigns["campaign_budget"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Attach campaign details to each session
session_campaigns = pd.merge(
    sessions[["customer_id", "campaign_id"]],
    campaigns[["campaign_id", "campaign_type", "campaign_budget"]],
    on="campaign_id",
    how="left"
)

campaign_data = session_campaigns.groupby("customer_id").agg(
    unique_campaigns=("campaign_id", "nunique"),
    avg_campaign_budget=("campaign_budget", "mean"),
    unique_campaign_types=("campaign_type", "nunique")
).reset_index()

print("Campaign Data Shape:", campaign_data.shape)
print("Unique Customer IDs:", campaign_data["customer_id"].nunique())
print("Missing Values:", campaign_data.isnull().sum().sum())
print(campaign_data.head())

#=======================
#       Campaign Budget Verification
#=======================

print("Original Campaign Budget Summary:")
print(campaigns["campaign_budget"].describe())

print("\nFirst 10 Campaign Budgets:")
print(campaigns["campaign_budget"].head(10))

print("\nUnique Campaign Count:", campaigns["campaign_id"].nunique())

#=======================
#       Final Feature Integration
#=======================

final_data = customer_data.copy()

final_data = pd.merge(
    final_data, session_data, on="customer_id", how="left"
)

final_data = pd.merge(
    final_data, transaction_data, on="customer_id", how="left"
)

final_data = pd.merge(
    final_data, customer_region_data, on="customer_id", how="left"
)

final_data = pd.merge(
    final_data, traffic_data, on="customer_id", how="left"
)

final_data = pd.merge(
    final_data, campaign_data, on="customer_id", how="left"
)

print("Final Dataset Shape:", final_data.shape)
print("Unique Customer IDs:", final_data["customer_id"].nunique())
print("Duplicate Customer IDs:", final_data["customer_id"].duplicated().sum())
print("Missing Values:", final_data.isnull().sum().sum())

print("\nMissing Values by Column:")
print(final_data.isnull().sum()[final_data.isnull().sum() > 0])

#=======================
#       Final Dataset Data Type Check
#=======================

print("Data Types:")
print(final_data.dtypes)

print("\nNumeric Columns:", len(final_data.select_dtypes(include="number").columns))
print("Object Columns:", len(final_data.select_dtypes(include="object").columns))

print("\nObject Column Names:")
print(final_data.select_dtypes(include="object").columns.tolist())

#=======================
#       Data Type Cleaning
#=======================

# Convert age to numeric
final_data["age"] = pd.to_numeric(
    final_data["age"], errors="coerce"
)

# Convert credit limit to numeric
final_data["credit_limit"] = (
    final_data["credit_limit"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

final_data["credit_limit"] = pd.to_numeric(
    final_data["credit_limit"], errors="coerce"
)

# Convert signup date to datetime
final_data["signup_date"] = pd.to_datetime(
    final_data["signup_date"], errors="coerce"
)

print("Updated Data Types:")
print(final_data[["age", "credit_limit", "signup_date"]].dtypes)

print("\nMissing Values After Conversion:")
print(final_data[["age", "credit_limit", "signup_date"]].isnull().sum())

#=======================
#       Invalid Age Investigation
#=======================

customers = pd.read_csv(path + "/customers.csv")

age_numeric = pd.to_numeric(
    customers["age"], errors="coerce"
)

invalid_age = customers.loc[
    age_numeric.isnull(), "age"
]

print("Invalid Age Count:", len(invalid_age))
print("\nInvalid Age Examples:")
print(invalid_age.value_counts().head(10))

print("\nValid Age Summary:")
print(age_numeric.describe())
#=======================
#       Age Missing Value Treatment
#=======================

age_median = final_data["age"].median()

final_data["age"] = final_data["age"].fillna(
    age_median
)

print("Age Median Used:", age_median)
print("Missing Ages After Treatment:",
      final_data["age"].isnull().sum())

#=======================
#       Email Open Rate Missing Value Treatment
#=======================

email_median = final_data["email_open_rate"].median()

final_data["email_open_rate"] = final_data[
    "email_open_rate"
].fillna(email_median)

print("Email Open Rate Median:", email_median)
print(
    "Missing Email Open Rates:",
    final_data["email_open_rate"].isnull().sum()
)

#=======================
#       Missing Activity Feature Treatment
#=======================

# Fill transaction features for customers with no transactions
transaction_columns = [
    "total_transactions",
    "total_spent",
    "total_items"
]

final_data[transaction_columns] = final_data[
    transaction_columns
].fillna(0)

# Fill session count and total features for customers with no sessions
session_columns = [
    "total_sessions",
    "total_pages_viewed",
    "total_cart_additions"
]

final_data[session_columns] = final_data[
    session_columns
].fillna(0)

print("Missing Transaction Features:")
print(final_data[transaction_columns].isnull().sum())

print("\nMissing Session Count/Total Features:")
print(final_data[session_columns].isnull().sum())

#=======================
#       Average Feature Missing Value Treatment
#=======================

average_columns = [
    "avg_session_duration",
    "avg_pages_viewed",
    "avg_cart_additions",
    "avg_order_value",
    "max_order_value",
    "avg_items_per_order"
]

print("Missing Values Before Treatment:")
print(final_data[average_columns].isnull().sum())

for column in average_columns:
    final_data[column] = final_data[column].fillna(
        final_data[column].median()
    )

print("\nMissing Values After Treatment:")
print(final_data[average_columns].isnull().sum())

#=======================
#       Geography and Engagement Missing Values
#=======================

numeric_columns = [
    "average_income",
    "urban_ratio",
    "internet_penetration",
    "unique_traffic_sources",
    "unique_device_types",
    "avg_bounce_flag",
    "unique_campaigns",
    "avg_campaign_budget",
    "unique_campaign_types"
]

for column in numeric_columns:
    final_data[column] = final_data[column].fillna(
        final_data[column].median()
    )

categorical_columns = [
    "geo_ip_region",
    "region_tier"
]

for column in categorical_columns:
    final_data[column] = final_data[column].fillna(
        final_data[column].mode()[0]
    )

print("Missing Values After Treatment:",
      final_data.isnull().sum().sum())

print("\nColumns Still Missing Values:")
print(final_data.isnull().sum()[final_data.isnull().sum() > 0])

#=======================
#       Categorical Value Standardization
#=======================

# Standardize country codes
final_data["country_code"] = (
    final_data["country_code"]
    .str.strip()
    .str.upper()
    .replace({"U.S.": "USA"})
)

# Standardize region and customer tier
final_data["region"] = final_data["region"].str.strip()
final_data["customer_tier"] = (
    final_data["customer_tier"].str.strip().str.title()
)

# Standardize geographic region tier
final_data["region_tier"] = (
    final_data["region_tier"].str.strip().str.title()
)

print("Country Codes:")
print(final_data["country_code"].value_counts())

print("\nCustomer Tiers:")
print(final_data["customer_tier"].value_counts())

print("\nGeographic Region Tiers:")
print(final_data["region_tier"].value_counts())

#=======================
#       Country Code Investigation
#=======================

print("Original Country Code Counts:")
print(customers["country_code"].value_counts())

print("\nStandardized Country Code Counts:")
print(final_data["country_code"].value_counts())

print("\nReferral Code Unique Count:")
print(final_data["referral_code"].nunique())

print("\nReferral Code Examples:")
print(final_data["referral_code"].value_counts().head(10))

#=======================
#       Country Code Verification
#=======================

print("Original US-related values:")
print(
    customers[
        customers["country_code"].str.strip().str.upper().isin(
            ["US", "USA", "U.S."]
        )
    ]["country_code"].value_counts()
)

print("\nCustomer counts by standardized country:")
print(final_data["country_code"].value_counts())

#=======================
#       Country Code Standardization Fix
#=======================

final_data["country_code"] = (
    final_data["country_code"]
    .str.strip()
    .str.upper()
    .replace({
        "US": "USA",
        "U.S.": "USA"
    })
)

print("Corrected Country Code Counts:")
print(final_data["country_code"].value_counts())

#=======================
#       High-Cardinality Feature Handling
#=======================

final_data = final_data.drop(
    columns=["referral_code"]
)

print("Dataset Shape After Dropping Referral Code:",
      final_data.shape)

print("Referral Code Exists:",
      "referral_code" in final_data.columns)
#=======================
#       Signup Date Feature Engineering
#=======================

final_data["signup_year"] = (
    final_data["signup_date"].dt.year
)

final_data["signup_month"] = (
    final_data["signup_date"].dt.month
)

final_data = final_data.drop(
    columns=["signup_date"]
)

print("Signup Date Features:")
print(final_data[["signup_year", "signup_month"]].head())

print("\nDataset Shape:", final_data.shape)
print("Signup Date Exists:", "signup_date" in final_data.columns)

#=======================
#       Categorical Feature Review
#=======================

categorical_columns = final_data.select_dtypes(
    include=["object"]
).columns.tolist()

print("Categorical Columns:", categorical_columns)

for column in categorical_columns:
    print(f"\n{column}: {final_data[column].nunique()} unique values")
    print(final_data[column].value_counts().head(10))
    
#=======================
#       Categorical Feature Encoding
#=======================

categorical_columns = [
    "country_code",
    "region",
    "customer_tier",
    "geo_ip_region",
    "region_tier"
]

final_data = pd.get_dummies(
    final_data,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print("Dataset Shape After Encoding:", final_data.shape)

print("\nRemaining Object Columns:")
print(
    final_data.select_dtypes(
        include=["object", "datetime"]
    ).columns.tolist()
)

#=======================
#       Numerical Outlier Analysis
#=======================

numeric_data = final_data.drop(
    columns=["customer_id", "target"]
).select_dtypes(include=["number"])

outlier_summary = []

for column in numeric_data.columns:
    Q1 = numeric_data[column].quantile(0.25)
    Q3 = numeric_data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outlier_count = (
        (numeric_data[column] < lower_limit) |
        (numeric_data[column] > upper_limit)
    ).sum()

    outlier_summary.append(
        [column, lower_limit, upper_limit, outlier_count]
    )

outlier_df = pd.DataFrame(
    outlier_summary,
    columns=[
        "Feature",
        "Lower Limit",
        "Upper Limit",
        "Outlier Count"
    ]
)

print("Features with the Most Potential Outliers:")
print(
    outlier_df.sort_values(
        by="Outlier Count",
        ascending=False
    ).head(15).to_string(index=False)
)

#=======================
#       Continuous Feature Outlier Analysis
#=======================

continuous_columns = [
    "age",
    "loyalty_score",
    "email_open_rate",
    "discount_usage_rate",
    "avg_review_score",
    "credit_limit",
    "avg_session_duration",
    "avg_pages_viewed",
    "avg_cart_additions",
    "avg_order_value",
    "max_order_value",
    "average_income",
    "urban_ratio",
    "internet_penetration",
    "avg_bounce_flag",
    "avg_campaign_budget"
]

outlier_summary = []

for column in continuous_columns:
    Q1 = final_data[column].quantile(0.25)
    Q3 = final_data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outlier_count = (
        (final_data[column] < lower_limit) |
        (final_data[column] > upper_limit)
    ).sum()

    outlier_summary.append(
        [column, lower_limit, upper_limit, outlier_count]
    )

outlier_df = pd.DataFrame(
    outlier_summary,
    columns=[
        "Feature",
        "Lower Limit",
        "Upper Limit",
        "Outlier Count"
    ]
)

print(
    outlier_df.sort_values(
        by="Outlier Count",
        ascending=False
    ).to_string(index=False)
)

#=======================
#       Feature Correlation Analysis
#=======================

numeric_data = final_data.drop(
    columns=["customer_id", "target"]
).select_dtypes(include=["number"])

correlation = numeric_data.corr().abs()

# Keep only the upper triangle to avoid duplicate pairs
upper_triangle = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))

high_correlations = (
    upper_triangle.stack()
    .reset_index()
)

high_correlations.columns = [
    "Feature 1",
    "Feature 2",
    "Correlation"
]

high_correlations = high_correlations[
    high_correlations["Correlation"] >= 0.90
].sort_values(
    by="Correlation",
    ascending=False
)

print("Highly Correlated Feature Pairs (|r| >= 0.90):")
print(high_correlations.to_string(index=False))

#=======================
#       Target and Feature Separation
#=======================

X = final_data.drop(
    columns=["customer_id", "target"]
)

y = final_data["target"]

print("Feature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Proportion:")
print((y.value_counts(normalize=True) * 100).round(2))

print("\nMissing Values in X:", X.isnull().sum().sum())
print("Missing Values in y:", y.isnull().sum())

#=======================
#       Train Validation Split
#=======================

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train Shape:", X_train.shape)
print("X_val Shape:", X_val.shape)

print("\nTraining Target Distribution:")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nValidation Target Distribution:")
print((y_val.value_counts(normalize=True) * 100).round(2))

#=======================
#       Feature Scaling
#=======================

from sklearn.preprocessing import StandardScaler

scale_columns = [
    "age",
    "loyalty_score",
    "email_open_rate",
    "discount_usage_rate",
    "avg_review_score",
    "credit_limit",
    "total_sessions",
    "avg_session_duration",
    "total_pages_viewed",
    "avg_pages_viewed",
    "total_cart_additions",
    "avg_cart_additions",
    "total_transactions",
    "total_spent",
    "avg_order_value",
    "max_order_value",
    "total_items",
    "avg_items_per_order",
    "average_income",
    "urban_ratio",
    "internet_penetration",
    "unique_traffic_sources",
    "unique_device_types",
    "avg_bounce_flag",
    "unique_campaigns",
    "avg_campaign_budget",
    "unique_campaign_types"
]

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

X_train_scaled[scale_columns] = scaler.fit_transform(
    X_train[scale_columns]
)

X_val_scaled[scale_columns] = scaler.transform(
    X_val[scale_columns]
)

print("Training Shape:", X_train_scaled.shape)
print("Validation Shape:", X_val_scaled.shape)

print("\nScaled Training Feature Means:")
print(X_train_scaled[scale_columns].mean().round(3))

print("\nMissing Values:")
print("Training:", X_train_scaled.isnull().sum().sum())
print("Validation:", X_val_scaled.isnull().sum().sum())

#=======================
#       Initial Logistic Regression Model
#=======================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_val_scaled)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))



#=======================
#       Logistic Regression Solver Test
#=======================

model = LogisticRegression(
    class_weight="balanced",
    max_iter=3000,
    solver="saga",
    random_state=42,
    tol=0.001
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_val_scaled)

print("Iterations Used:", model.n_iter_)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

#=======================
#       Leakage-Free Pipeline Preparation
#=======================

# Load original customer data and training labels
customers_raw = pd.read_csv(path + "/customers.csv")
train_raw = pd.read_csv(path + "/train.csv")

# Keep only training customers
raw_data = pd.merge(
    customers_raw,
    train_raw,
    on="customer_id",
    how="inner"
)

# Separate target and features
X_raw = raw_data.drop(
    columns=["customer_id", "target", "referral_code"]
)

y_raw = raw_data["target"]

print("Raw Feature Matrix Shape:", X_raw.shape)
print("Target Shape:", y_raw.shape)

print("\nRaw Feature Data Types:")
print(X_raw.dtypes)

print("\nMissing Values:")
print(X_raw.isnull().sum()[X_raw.isnull().sum() > 0])

#=======================
#       Raw Session Feature Engineering
#=======================

sessions_raw = pd.read_csv(path + "/sessions.csv")

# Standardize category values
sessions_raw["traffic_source"] = (
    sessions_raw["traffic_source"].str.strip().str.lower()
)

sessions_raw["device_type"] = (
    sessions_raw["device_type"].str.strip().str.lower()
)

sessions_raw["bounce_flag"] = (
    sessions_raw["bounce_flag"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "1": 1,
        "true": 1,
        "yes": 1,
        "0": 0,
        "false": 0,
        "no": 0
    })
)

session_features_raw = sessions_raw.groupby("customer_id").agg(
    total_sessions=("session_id", "count"),
    avg_session_duration=("session_duration", "mean"),
    total_pages_viewed=("pages_viewed", "sum"),
    avg_pages_viewed=("pages_viewed", "mean"),
    total_cart_additions=("cart_additions", "sum"),
    avg_cart_additions=("cart_additions", "mean"),
    unique_traffic_sources=("traffic_source", "nunique"),
    unique_device_types=("device_type", "nunique"),
    avg_bounce_flag=("bounce_flag", "mean")
).reset_index()

print("Session Feature Shape:", session_features_raw.shape)
print("Missing Values:")
print(
    session_features_raw.isnull().sum()[
        session_features_raw.isnull().sum() > 0
    ]
)
print(session_features_raw.head())

#=======================
#       Raw Transaction Feature Engineering
#=======================

transactions_raw = pd.read_csv(path + "/transactions.csv")

# Convert order value to numeric
transactions_raw["order_value"] = (
    transactions_raw["order_value"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

transactions_raw["order_value"] = pd.to_numeric(
    transactions_raw["order_value"],
    errors="coerce"
)

transaction_features_raw = transactions_raw.groupby(
    "customer_id"
).agg(
    total_transactions=("transaction_id", "count"),
    total_spent=("order_value", "sum"),
    avg_order_value=("order_value", "mean"),
    max_order_value=("order_value", "max"),
    total_items=("items_count", "sum"),
    avg_items_per_order=("items_count", "mean")
).reset_index()

print("Transaction Feature Shape:", transaction_features_raw.shape)

print("\nMissing Values:")
print(
    transaction_features_raw.isnull().sum()[
        transaction_features_raw.isnull().sum() > 0
    ]
)

print(transaction_features_raw.head())

#=======================
#       Raw Geographic Feature Engineering
#=======================

geo_data_raw = pd.read_csv(path + "/geo_data.csv")

# Find each customer's most frequently visited region
customer_region_raw = (
    sessions_raw.groupby(["customer_id", "geo_ip_region"])
    .size()
    .reset_index(name="region_visits")
)

customer_region_raw = customer_region_raw.sort_values(
    ["customer_id", "region_visits"],
    ascending=[True, False]
)

customer_region_raw = customer_region_raw.drop_duplicates(
    subset="customer_id"
)

customer_region_raw = customer_region_raw[
    ["customer_id", "geo_ip_region"]
]

# Attach geographic information
geo_features_raw = pd.merge(
    customer_region_raw,
    geo_data_raw,
    on="geo_ip_region",
    how="left"
)

print("Geographic Feature Shape:", geo_features_raw.shape)
print("Unique Customers:", geo_features_raw["customer_id"].nunique())

print("\nMissing Values:")
print(
    geo_features_raw.isnull().sum()[
        geo_features_raw.isnull().sum() > 0
    ]
)

print(geo_features_raw.head())
#=======================
#       Raw Traffic and Campaign Feature Engineering
#=======================

campaigns_raw = pd.read_csv(path + "/marketing_campaigns.csv")

# Convert campaign budget to numeric
campaigns_raw["campaign_budget"] = (
    campaigns_raw["campaign_budget"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

campaigns_raw["campaign_budget"] = pd.to_numeric(
    campaigns_raw["campaign_budget"],
    errors="coerce"
)

# Join campaign information to sessions
session_campaigns_raw = pd.merge(
    sessions_raw[
        ["customer_id", "campaign_id", "traffic_source", "device_type"]
    ],
    campaigns_raw[
        ["campaign_id", "campaign_type", "campaign_budget"]
    ],
    on="campaign_id",
    how="left"
)

traffic_campaign_features_raw = session_campaigns_raw.groupby(
    "customer_id"
).agg(
    unique_traffic_sources=("traffic_source", "nunique"),
    unique_device_types=("device_type", "nunique"),
    unique_campaigns=("campaign_id", "nunique"),
    avg_campaign_budget=("campaign_budget", "mean"),
    unique_campaign_types=("campaign_type", "nunique")
).reset_index()

print("Traffic/Campaign Feature Shape:",
      traffic_campaign_features_raw.shape)

print("Unique Customers:",
      traffic_campaign_features_raw["customer_id"].nunique())

print("\nMissing Values:")
print(
    traffic_campaign_features_raw.isnull().sum()[
        traffic_campaign_features_raw.isnull().sum() > 0
    ]
)

print(traffic_campaign_features_raw.head())

#=======================
#       Leakage-Free Feature Integration
#=======================

model_data = X_raw.copy()
model_data["target"] = y_raw.values

model_data = pd.merge(
    model_data,
    session_features_raw,
    on="customer_id",
    how="left"
) if "customer_id" in model_data.columns else model_data

#=======================
#       Leakage-Free Feature Integration
#=======================

model_data = raw_data.copy()

model_data = pd.merge(
    model_data,
    session_features_raw,
    on="customer_id",
    how="left"
)

model_data = pd.merge(
    model_data,
    transaction_features_raw,
    on="customer_id",
    how="left"
)

model_data = pd.merge(
    model_data,
    geo_features_raw,
    on="customer_id",
    how="left"
)

model_data = pd.merge(
    model_data,
    traffic_campaign_features_raw,
    on="customer_id",
    how="left"
)

print("Integrated Dataset Shape:", model_data.shape)
print("Unique Customer IDs:", model_data["customer_id"].nunique())
print("Duplicate Customer IDs:", model_data["customer_id"].duplicated().sum())

print("\nMissing Values by Column:")
print(
    model_data.isnull().sum()[
        model_data.isnull().sum() > 0
    ]
)

#=======================
#       Duplicate Feature Cleanup
#=======================

# Keep the session-level versions and remove duplicate versions
model_data = model_data.drop(
    columns=[
        "unique_traffic_sources_y",
        "unique_device_types_y"
    ]
)

# Rename session-level versions clearly
model_data = model_data.rename(
    columns={
        "unique_traffic_sources_x": "unique_traffic_sources",
        "unique_device_types_x": "unique_device_types"
    }
)

print("Dataset Shape After Cleanup:", model_data.shape)

print("\nDuplicate Column Names:",
      model_data.columns[model_data.columns.duplicated()].tolist())

print("\nTraffic/Device Columns:")
print([
    column for column in model_data.columns
    if "traffic_sources" in column or "device_types" in column
])

#=======================
#       Pipeline Feature Preparation
#=======================

X_raw = model_data.drop(
    columns=["customer_id", "target"]
)

y_raw = model_data["target"]

# Identify categorical and numerical columns
categorical_columns = X_raw.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X_raw.select_dtypes(
    include=["number"]
).columns.tolist()

print("Feature Matrix Shape:", X_raw.shape)
print("Target Shape:", y_raw.shape)

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumeric Column Count:", len(numeric_columns))
print(numeric_columns)

print("\nMissing Values in Features:", X_raw.isnull().sum().sum())

#=======================
#       Raw Feature Type Preparation
#=======================

# Convert age to numeric
X_raw["age"] = pd.to_numeric(
    X_raw["age"], errors="coerce"
)

# Convert credit limit to numeric
X_raw["credit_limit"] = (
    X_raw["credit_limit"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

X_raw["credit_limit"] = pd.to_numeric(
    X_raw["credit_limit"], errors="coerce"
)

# Extract signup date features
X_raw["signup_date"] = pd.to_datetime(
    X_raw["signup_date"], errors="coerce"
)

X_raw["signup_year"] = X_raw["signup_date"].dt.year
X_raw["signup_month"] = X_raw["signup_date"].dt.month

# Remove raw date and high-cardinality identifier
X_raw = X_raw.drop(
    columns=["signup_date", "referral_code"]
)

print("Prepared Feature Shape:", X_raw.shape)

print("\nRemaining Object Columns:")
print(X_raw.select_dtypes(include=["object"]).columns.tolist())

print("\nMissing Values:")
print(X_raw.isnull().sum()[X_raw.isnull().sum() > 0])

#=======================
#       Train-Validation Split
#=======================

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_raw,
    y_raw,
    test_size=0.20,
    random_state=42,
    stratify=y_raw
)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts(normalize=True))

print("\nValidation Target Distribution:")
print(y_val.value_counts(normalize=True))

#=======================
#       Identify Feature Types
#=======================

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

print("Number of Numeric Features:", len(numeric_columns))
print("Number of Categorical Features:", len(categorical_columns))

print("\nNumeric Features:")
print(numeric_columns)

print("\nCategorical Features:")
print(categorical_columns)

#=======================
#       Preprocessing Pipeline
#=======================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Numeric features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    ))
])

# Combine both transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns)
    ]
)

print("Preprocessing pipeline created successfully.")

#=======================
#       Apply Preprocessing
#=======================

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

print("Processed Training Shape:", X_train_processed.shape)
print("Processed Validation Shape:", X_val_processed.shape)

#=======================
#       Verify Processed Data
#=======================

import numpy as np
from scipy import sparse

print("Processed Training Shape:", X_train_processed.shape)
print("Processed Validation Shape:", X_val_processed.shape)

if sparse.issparse(X_train_processed):
    train_missing = np.isnan(X_train_processed.data).sum()
else:
    train_missing = np.isnan(X_train_processed).sum()

if sparse.issparse(X_val_processed):
    val_missing = np.isnan(X_val_processed.data).sum()
else:
    val_missing = np.isnan(X_val_processed).sum()

print("\nMissing Values in Training Data:", train_missing)
print("Missing Values in Validation Data:", val_missing)

#=======================
#       Baseline Model Training
#=======================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_processed, y_train)

y_pred = model.predict(X_val_processed)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, zero_division=0))
print("Recall:", recall_score(y_val, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_val, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))