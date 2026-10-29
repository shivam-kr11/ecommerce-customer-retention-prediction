# ============================================================================
# E-COMMERCE CUSTOMER RETENTION AND CHURN PREDICTION
#
# WORKFLOW:
# 1. Load and inspect each dataset.
# 2. Clean each dataset one at a time.
# 3. Create customer-level features and combine the datasets.
# 4. Split customers into training and validation groups.
# 5. Encode, scale, train, compare, and save the models.
# ============================================================================

#================ IMPORTING LIBRARIES ================
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


#================ CONFIGURATION ================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
FIGURES_PATH = BASE_DIR / "reports" / "figures"
MODEL_PATH = BASE_DIR / "models" / "retention_model.pkl"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
TOP_INPUT_FEATURE_COUNT = 8


#================ HELPER: INSPECT A DATASET ================
def inspect_dataset(data, dataset_name):
    """Print the same basic checks for each raw dataset."""
    print("\n" + "=" * 60)
    print("DATASET:", dataset_name)
    print("=" * 60)
    print("Shape:", data.shape)
    print("Columns:", data.columns.tolist())
    print("Data types:")
    print(data.dtypes)
    print("Missing values:")
    print(data.isnull().sum())
    print("Duplicate rows:", data.duplicated().sum())
    print("First 3 rows:")
    print(data.head(3))


#================ HELPER: CLEAN CURRENCY VALUES ================
def clean_currency(data, column):
    """Remove currency symbols and commas, then convert the column to numbers."""
    values = data[column].astype("string")
    values = values.str.replace("$", "", regex=False)
    values = values.str.replace(",", "", regex=False)
    data[column] = pd.to_numeric(values, errors="coerce")
    return data


#================ HELPER: CLEAN DATE VALUES ================
def clean_date(data, column):
    """Parse dates written as day-month-year."""
    data[column] = pd.to_datetime(data[column], dayfirst=True, errors="coerce")
    return data


#================ HELPER: VERIFY A CLEANED DATASET ================
def verify_clean_dataset(data, dataset_name):
    """Show the final missing-value and duplicate checks for one table."""
    print("\nFinal check:", dataset_name)
    print("Shape:", data.shape)
    print("Missing values:", data.isnull().sum().sum())
    print("Duplicate rows:", data.duplicated().sum())


#================ HELPER: SAVE A GRAPH ================
def save_graph(file_name):
    """Save the current graph and show it when an interactive display exists."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    graph_path = FIGURES_PATH / file_name
    plt.tight_layout()
    plt.savefig(graph_path, dpi=200, bbox_inches="tight")
    print("Graph saved:", graph_path)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


#================ LOADING DATASETS ================
print("Loading raw datasets...")

customers = pd.read_csv(DATA_PATH / "customers.csv")
sessions = pd.read_csv(DATA_PATH / "sessions.csv")
transactions = pd.read_csv(DATA_PATH / "transactions.csv")
geo_data = pd.read_csv(DATA_PATH / "geo_data.csv")
marketing = pd.read_csv(DATA_PATH / "marketing_campaigns.csv")
train_data = pd.read_csv(DATA_PATH / "train.csv")
test_data = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")


#================ OBJECTIVE 1: INSPECT EACH DATASET ================
inspect_dataset(customers, "customers.csv")
inspect_dataset(sessions, "sessions.csv")
inspect_dataset(transactions, "transactions.csv")
inspect_dataset(geo_data, "geo_data.csv")
inspect_dataset(marketing, "marketing_campaigns.csv")
inspect_dataset(train_data, "train.csv")
inspect_dataset(test_data, "test.csv")
inspect_dataset(sample_submission, "sample_submission.csv")

print("\nTarget distribution:")
print(train_data["target"].value_counts())
print("Target percentages:")
print((train_data["target"].value_counts(normalize=True) * 100).round(2))


#================ OBJECTIVE 2: CLEAN LABEL FILES AND MAKE VALIDATION SPLIT ================
# This split is created before any source table is imputed or used for feature selection.
print("\n================ PREPARING TRAIN AND TEST LABELS ================")
train_data = train_data.drop_duplicates(subset="customer_id").copy()
test_data = test_data.drop_duplicates(subset="customer_id").copy()
train_data["target"] = pd.to_numeric(train_data["target"], errors="coerce")
train_data = train_data.dropna(subset=["customer_id", "target"])
train_data["target"] = train_data["target"].astype(int)

sample_submission = sample_submission.drop_duplicates(subset="customer_id").copy()
sample_submission["target"] = pd.to_numeric(
    sample_submission["target"], errors="coerce"
)
sample_submission = sample_submission.dropna(subset=["customer_id", "target"])
sample_submission["target"] = sample_submission["target"].astype(int)

train_customer_ids, validation_customer_ids = train_test_split(
    train_data["customer_id"],
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=train_data["target"],
)
train_customer_set = set(train_customer_ids)
validation_customer_set = set(validation_customer_ids)
print("Training customers:", len(train_customer_set))
print("Validation customers:", len(validation_customer_set))
if len(sessions) < 2_000_000:
    print(
        "WARNING: sessions.csv has", len(sessions),
        "rows; the project documentation says 2,000,000.",
    )


#================ OBJECTIVE 2: CLEAN CUSTOMERS DATASET ================
print("\n================ CLEANING CUSTOMERS DATASET ================")
customers = customers.drop_duplicates(subset="customer_id").copy()
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers = clean_currency(customers, "credit_limit")
customers = clean_date(customers, "signup_date")
customers["country_code"] = customers["country_code"].astype("string").str.strip().str.upper()
for column in ["region", "customer_tier", "referral_code"]:
    customers[column] = customers[column].astype("string").str.strip()

customer_train_rows = customers["customer_id"].isin(train_customer_set)
for column in [
    "age", "credit_limit", "email_open_rate", "loyalty_score",
    "discount_usage_rate", "avg_review_score",
]:
    customers[column] = customers[column].fillna(
        customers.loc[customer_train_rows, column].median()
    )
for column in ["country_code", "region", "customer_tier", "referral_code"]:
    customers[column] = customers[column].fillna("Unknown")
verify_clean_dataset(customers, "customers.csv")


#================ OBJECTIVE 2: CLEAN SESSIONS DATASET ================
print("\n================ CLEANING SESSIONS DATASET ================")
sessions = sessions.drop_duplicates(subset="session_id").copy()
sessions = clean_date(sessions, "session_timestamp")
session_number_columns = ["session_duration", "pages_viewed", "cart_additions"]
for column in session_number_columns:
    sessions[column] = pd.to_numeric(sessions[column], errors="coerce")
sessions["bounce_flag"] = (
    sessions["bounce_flag"].astype("string").str.lower().isin(["1", "true", "yes"])
).astype(int)
for column in ["traffic_source", "device_type", "geo_ip_region"]:
    sessions[column] = sessions[column].astype("string").str.strip()

session_train_rows = sessions["customer_id"].isin(train_customer_set)
for column in session_number_columns:
    sessions[column] = sessions[column].fillna(
        sessions.loc[session_train_rows, column].median()
    )
for column in ["traffic_source", "device_type", "geo_ip_region"]:
    sessions[column] = sessions[column].fillna("Unknown")
session_cap = sessions.loc[session_train_rows, "session_duration"].quantile(0.99)
sessions["session_duration"] = sessions["session_duration"].clip(upper=session_cap)
verify_clean_dataset(sessions, "sessions.csv")


#================ OBJECTIVE 2: CLEAN TRANSACTIONS DATASET ================
print("\n================ CLEANING TRANSACTIONS DATASET ================")
transactions = transactions.drop_duplicates(subset="transaction_id").copy()
transactions = clean_date(transactions, "transaction_timestamp")
transactions = clean_currency(transactions, "order_value")
transactions["items_count"] = pd.to_numeric(transactions["items_count"], errors="coerce")
transactions["discount_applied"] = (
    transactions["discount_applied"].astype("string").str.lower().isin(["1", "true", "yes"])
).astype(int)
for column in ["payment_method", "shipping_speed", "high_value_flag"]:
    transactions[column] = transactions[column].astype("string").str.strip()

transaction_train_rows = transactions["customer_id"].isin(train_customer_set)
for column in ["order_value", "items_count"]:
    transactions[column] = transactions[column].fillna(
        transactions.loc[transaction_train_rows, column].median()
    )
for column in ["payment_method", "shipping_speed", "high_value_flag"]:
    transactions[column] = transactions[column].fillna("Unknown")
order_cap = transactions.loc[transaction_train_rows, "order_value"].quantile(0.99)
transactions["order_value"] = transactions["order_value"].clip(upper=order_cap)
verify_clean_dataset(transactions, "transactions.csv")


#================ OBJECTIVE 2: CLEAN GEO DATASET ================
print("\n================ CLEANING GEO DATASET ================")
geo_data = geo_data.drop_duplicates(subset="geo_ip_region").copy()
for column in ["geo_ip_region", "region_tier"]:
    geo_data[column] = geo_data[column].astype("string").str.strip()
for column in ["average_income", "urban_ratio", "internet_penetration"]:
    geo_data[column] = pd.to_numeric(geo_data[column], errors="coerce")
    geo_data[column] = geo_data[column].fillna(geo_data[column].median())
geo_data["region_tier"] = geo_data["region_tier"].fillna("Unknown")
verify_clean_dataset(geo_data, "geo_data.csv")


#================ OBJECTIVE 2: CLEAN MARKETING DATASET ================
print("\n================ CLEANING MARKETING DATASET ================")
marketing = marketing.drop_duplicates(subset="campaign_id").copy()
marketing = clean_currency(marketing, "campaign_budget")
marketing = clean_date(marketing, "start_date")
marketing = clean_date(marketing, "end_date")
for column in ["campaign_type", "region_target"]:
    marketing[column] = marketing[column].astype("string").str.strip()
marketing["campaign_budget"] = marketing["campaign_budget"].fillna(
    marketing["campaign_budget"].median()
)
for column in ["campaign_type", "region_target"]:
    marketing[column] = marketing[column].fillna("Unknown")
verify_clean_dataset(marketing, "marketing_campaigns.csv")


#================ OBJECTIVE 2: VERIFY LABEL AND SUBMISSION DATASETS ================
print("\n================ VERIFYING LABEL AND SUBMISSION DATASETS ================")
verify_clean_dataset(train_data, "train.csv")
verify_clean_dataset(test_data, "test.csv")
verify_clean_dataset(sample_submission, "sample_submission.csv")
print("Target distribution:")
print(train_data["target"].value_counts())
print("Target percentages:")
print((train_data["target"].value_counts(normalize=True) * 100).round(2))

plt.figure(figsize=(6, 4))
sns.countplot(data=train_data, x="target")
plt.title("Training Target Distribution")
plt.xlabel("Target (0 = Retained, 1 = Churned)")
plt.ylabel("Customer Count")
save_graph("01_target_distribution.png")


#================ OBJECTIVE 2: SAVE CLEANED DATASETS ================
print("\n================ SAVING CLEANED DATASETS ================")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
cleaned_datasets = {
    "customers.csv": customers,
    "sessions.csv": sessions,
    "transactions.csv": transactions,
    "geo_data.csv": geo_data,
    "marketing_campaigns.csv": marketing,
    "train.csv": train_data,
    "test.csv": test_data,
    "sample_submission.csv": sample_submission,
}
for file_name, data in cleaned_datasets.items():
    output_path = PROCESSED_PATH / file_name
    data.to_csv(output_path, index=False)
    print("Saved:", output_path)


#================ OBJECTIVE 3: PLOT CLEANED FEATURE DISTRIBUTIONS ================
print("\n================ PLOTTING CLEANED FEATURE DISTRIBUTIONS ================")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(customers["age"], bins=30, ax=axes[0, 0])
axes[0, 0].set_title("Customer Age")
sns.histplot(customers["loyalty_score"], bins=30, ax=axes[0, 1])
axes[0, 1].set_title("Customer Loyalty Score")
sns.histplot(sessions["session_duration"].sample(
    n=min(100000, len(sessions)), random_state=RANDOM_STATE
), bins=30, ax=axes[1, 0])
axes[1, 0].set_title("Session Duration (Sample)")
sns.histplot(transactions["order_value"].sample(
    n=min(100000, len(transactions)), random_state=RANDOM_STATE
), bins=30, ax=axes[1, 1])
axes[1, 1].set_title("Order Value (Sample)")
save_graph("02_cleaned_feature_distributions.png")


#================ OBJECTIVE 4: CREATE CUSTOMER PROFILE FEATURES ================
print("\n================ CREATING CUSTOMER PROFILE FEATURES ================")

reference_dates = [
    customers["signup_date"].max(),
    sessions["session_timestamp"].max(),
    transactions["transaction_timestamp"].max(),
]
reference_date = max(date for date in reference_dates if pd.notna(date))

customers["signup_year"] = customers["signup_date"].dt.year
customers["signup_month"] = customers["signup_date"].dt.month
customers["customer_tenure_days"] = (reference_date - customers["signup_date"]).dt.days

for column in ["signup_year", "signup_month", "customer_tenure_days"]:
    median_value = customers.loc[customer_train_rows, column].median()
    customers[column] = customers[column].fillna(median_value)

print("Observation reference date:", reference_date.date())
print(customers[["customer_id", "signup_year", "customer_tenure_days"]].head(3))


#================ OBJECTIVE 4: CREATE SESSION FEATURES ================
print("\n================ CREATING SESSION FEATURES ================")

sessions["days_since_session"] = (reference_date - sessions["session_timestamp"]).dt.days
sessions["sessions_last_30_days"] = (sessions["days_since_session"] <= 30).astype(int)
sessions["sessions_last_90_days"] = (sessions["days_since_session"] <= 90).astype(int)
sessions["sessions_last_180_days"] = (sessions["days_since_session"] <= 180).astype(int)
session_features = sessions.groupby("customer_id").agg(
    total_sessions=("session_id", "count"),
    avg_session_duration=("session_duration", "mean"),
    total_pages_viewed=("pages_viewed", "sum"),
    avg_pages_viewed=("pages_viewed", "mean"),
    total_cart_additions=("cart_additions", "sum"),
    avg_cart_additions=("cart_additions", "mean"),
    bounce_rate=("bounce_flag", "mean"),
    recency_session_days=("days_since_session", "min"),
    sessions_last_30_days=("sessions_last_30_days", "sum"),
    sessions_last_90_days=("sessions_last_90_days", "sum"),
    sessions_last_180_days=("sessions_last_180_days", "sum"),
).reset_index()

customer_region_counts = (
    sessions.groupby(["customer_id", "geo_ip_region"])
    .size()
    .reset_index(name="region_visits")
)
customer_primary_region = (
    customer_region_counts.sort_values(
        ["customer_id", "region_visits"], ascending=[True, False]
    )
    .drop_duplicates(subset="customer_id")[["customer_id", "geo_ip_region"]]
)

print("Session features:", session_features.shape)


#================ OBJECTIVE 4: CREATE TRANSACTION FEATURES ================
print("\n================ CREATING TRANSACTION FEATURES ================")

transactions["days_since_transaction"] = (
    reference_date - transactions["transaction_timestamp"]
).dt.days
transactions["transactions_last_30_days"] = (
    transactions["days_since_transaction"] <= 30
).astype(int)
transactions["transactions_last_90_days"] = (
    transactions["days_since_transaction"] <= 90
).astype(int)
transactions["transactions_last_180_days"] = (
    transactions["days_since_transaction"] <= 180
).astype(int)
transactions["order_value_last_90_days"] = transactions["order_value"].where(
    transactions["days_since_transaction"] <= 90, 0
)
transaction_features = transactions.groupby("customer_id").agg(
    total_transactions=("transaction_id", "count"),
    total_spent=("order_value", "sum"),
    avg_order_value=("order_value", "mean"),
    max_order_value=("order_value", "max"),
    total_items=("items_count", "sum"),
    avg_items_per_order=("items_count", "mean"),
    discount_trans_ratio=("discount_applied", "mean"),
    recency_trans_days=("days_since_transaction", "min"),
    transactions_last_30_days=("transactions_last_30_days", "sum"),
    transactions_last_90_days=("transactions_last_90_days", "sum"),
    transactions_last_180_days=("transactions_last_180_days", "sum"),
    order_value_last_90_days=("order_value_last_90_days", "sum"),
).reset_index()

print("Transaction features:", transaction_features.shape)


#================ OBJECTIVE 4: CREATE MARKETING FEATURES ================
print("\n================ CREATING MARKETING FEATURES ================")

session_campaigns = sessions[["customer_id", "campaign_id"]].merge(
    marketing[["campaign_id", "campaign_type", "campaign_budget"]],
    on="campaign_id",
    how="left",
)
campaign_features = session_campaigns.groupby("customer_id").agg(
    avg_campaign_budget=("campaign_budget", "mean"),
    total_campaign_budget=("campaign_budget", "sum"),
).reset_index()

customer_campaign_type = (
    session_campaigns.dropna(subset=["campaign_type"])
    .groupby(["customer_id", "campaign_type"])
    .size()
    .reset_index(name="campaign_visits")
    .sort_values(["customer_id", "campaign_visits"], ascending=[True, False])
    .drop_duplicates(subset="customer_id")
    [["customer_id", "campaign_type"]]
    .rename(columns={"campaign_type": "primary_campaign_type"})
)

print("Campaign features:", campaign_features.shape)


#================ OBJECTIVE 4: COMBINE DATASETS ================
print("\n================ COMBINING CUSTOMER DATASETS ================")

master_data = customers.copy()
master_data = master_data.merge(session_features, on="customer_id", how="left")
master_data = master_data.merge(transaction_features, on="customer_id", how="left")
master_data = master_data.merge(customer_primary_region, on="customer_id", how="left")
master_data = master_data.merge(campaign_features, on="customer_id", how="left")
master_data = master_data.merge(customer_campaign_type, on="customer_id", how="left")
master_data = master_data.merge(geo_data, on="geo_ip_region", how="left")

activity_columns = [
    "total_sessions", "avg_session_duration", "total_pages_viewed",
    "avg_pages_viewed", "total_cart_additions", "avg_cart_additions",
    "bounce_rate", "total_transactions", "total_spent", "avg_order_value",
    "max_order_value", "total_items", "avg_items_per_order",
    "discount_trans_ratio", "avg_campaign_budget", "total_campaign_budget",
    "sessions_last_30_days", "sessions_last_90_days", "sessions_last_180_days",
    "transactions_last_30_days", "transactions_last_90_days",
    "transactions_last_180_days", "order_value_last_90_days",
]
master_data[activity_columns] = master_data[activity_columns].fillna(0)
master_data["recency_session_days"] = master_data["recency_session_days"].fillna(9999)
master_data["recency_trans_days"] = master_data["recency_trans_days"].fillna(9999)
master_data["geo_ip_region"] = master_data["geo_ip_region"].fillna("Unknown")
master_data["region_tier"] = master_data["region_tier"].fillna("Unknown")
master_data["primary_campaign_type"] = master_data["primary_campaign_type"].fillna("Unknown")

# Add simple ratios that describe customer engagement and spending.
master_data["cart_to_order_ratio"] = (
    master_data["total_transactions"] / (master_data["total_cart_additions"] + 1)
)
master_data["spend_per_session"] = (
    master_data["total_spent"] / (master_data["total_sessions"] + 1)
)
master_data["pages_per_session"] = (
    master_data["total_pages_viewed"] / (master_data["total_sessions"] + 1)
)
master_data["spend_to_credit_ratio"] = (
    master_data["total_spent"] / (master_data["credit_limit"] + 1)
)
master_data["recent_spend_share_90_days"] = (
    master_data["order_value_last_90_days"] / (master_data["total_spent"] + 1)
)

for column in ["average_income", "urban_ratio", "internet_penetration"]:
    median_value = master_data.loc[
        master_data["customer_id"].isin(train_customer_set), column
    ].median()
    master_data[column] = master_data[column].fillna(median_value)

print("Combined customer data:", master_data.shape)
print("Unique customers:", master_data["customer_id"].nunique())
print("Remaining missing values:", master_data.isnull().sum().sum())


#================ OBJECTIVE 5: PLOT TRAINING FEATURE CORRELATIONS ================
# Use training customers only so validation labels do not influence this chart.
print("\n================ PLOTTING TRAINING CORRELATION MATRIX ================")
training_correlation_data = master_data[
    master_data["customer_id"].isin(train_customer_set)
].merge(train_data, on="customer_id", how="inner")
correlation_columns = training_correlation_data.select_dtypes(
    include="number"
).columns.drop("customer_id", errors="ignore")
correlation_matrix = training_correlation_data[correlation_columns].corr()
target_correlations = correlation_matrix["target"].drop("target")
top_correlation_features = target_correlations.abs().nlargest(12).index.tolist()
heatmap_columns = top_correlation_features + ["target"]

plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation_matrix.loc[heatmap_columns, heatmap_columns],
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.4,
)
plt.title("Training Feature Correlation Matrix (Top Features)")
save_graph("03_training_correlation_matrix.png")


#================ OBJECTIVE 5: PREPARE TRAIN AND VALIDATION DATA ================
print("\n================ PREPARING MODEL INPUTS ================")

labelled_data = master_data.merge(train_data, on="customer_id", how="inner")
training_rows = labelled_data["customer_id"].isin(train_customer_set)
validation_rows = labelled_data["customer_id"].isin(validation_customer_set)
training_data = labelled_data.loc[training_rows].copy()
validation_data = labelled_data.loc[validation_rows].copy()

drop_columns = [
    "customer_id", "target", "signup_date", "referral_code",
    "noise_customer_0", "noise_customer_1", "noise_customer_2",
    "noise_customer_3", "noise_customer_4", "pages_per_session", "signup_year",
]
categorical_columns = [
    "country_code", "region", "customer_tier", "geo_ip_region",
    "region_tier", "primary_campaign_type",
]

X_train_raw = training_data.drop(columns=drop_columns, errors="ignore")
y_train = training_data["target"]
X_validation_raw = validation_data.drop(columns=drop_columns, errors="ignore")
y_validation = validation_data["target"]

all_numeric_columns = [
    column for column in X_train_raw.columns if column not in categorical_columns
]

# Hold out part of the training split for threshold calibration. The final
# validation customers remain untouched until model evaluation.
model_fit_indices, threshold_indices = train_test_split(
    X_train_raw.index.to_numpy(),
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train,
)
model_fit_indices = pd.Index(model_fit_indices)
threshold_indices = pd.Index(threshold_indices)
y_model_fit = y_train.loc[model_fit_indices]
y_threshold = y_train.loc[threshold_indices]

# Encode using model-fit customers only. Calibration and final validation rows
# are transformed with this encoder and cannot influence category discovery.
initial_encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
initial_encoder.fit(X_train_raw.loc[model_fit_indices, categorical_columns])
initial_train_categories = initial_encoder.transform(
    X_train_raw[categorical_columns]
).toarray()
initial_validation_categories = initial_encoder.transform(
    X_validation_raw[categorical_columns]
).toarray()
initial_category_names = initial_encoder.get_feature_names_out(categorical_columns)

initial_train_category_data = pd.DataFrame(
    initial_train_categories,
    columns=initial_category_names,
    index=X_train_raw.index,
)
initial_validation_category_data = pd.DataFrame(
    initial_validation_categories,
    columns=initial_category_names,
    index=X_validation_raw.index,
)
initial_X_train = pd.concat(
    [X_train_raw[all_numeric_columns], initial_train_category_data], axis=1
)
initial_X_validation = pd.concat(
    [X_validation_raw[all_numeric_columns], initial_validation_category_data], axis=1
)

initial_scaler = StandardScaler()
initial_scaler.fit(initial_X_train.loc[model_fit_indices])
initial_X_train_scaled = initial_scaler.transform(initial_X_train)
model_fit_positions = initial_X_train.index.get_indexer(model_fit_indices)
threshold_positions = initial_X_train.index.get_indexer(threshold_indices)

# Use stratified folds to find source features that are repeatedly useful.
# Importance is averaged across folds instead of relying on one tree fit.
print("\n================ CROSS-VALIDATING FEATURE IMPORTANCE ================")
feature_importance = dict.fromkeys(all_numeric_columns + categorical_columns, 0.0)
feature_selection_frequency = dict.fromkeys(feature_importance, 0)
feature_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
fit_feature_matrix = initial_X_train_scaled[model_fit_positions]
fit_feature_labels = y_model_fit.to_numpy()

for fold_number, (fold_train_indices, _) in enumerate(
    feature_folds.split(fit_feature_matrix, fit_feature_labels), start=1
):
    feature_ranker = DecisionTreeClassifier(
        max_depth=6,
        random_state=RANDOM_STATE + fold_number,
        class_weight="balanced",
    )
    feature_ranker.fit(
        fit_feature_matrix[fold_train_indices],
        fit_feature_labels[fold_train_indices],
    )

    fold_importance = dict.fromkeys(feature_importance, 0.0)
    for encoded_name, importance in zip(
        initial_X_train.columns, feature_ranker.feature_importances_
    ):
        if encoded_name in fold_importance:
            fold_importance[encoded_name] += importance
        else:
            for column in sorted(categorical_columns, key=len, reverse=True):
                if encoded_name.startswith(column + "_"):
                    fold_importance[column] += importance
                    break

    for column, importance in fold_importance.items():
        feature_importance[column] += importance / 3
    fold_top_features = sorted(
        fold_importance, key=fold_importance.get, reverse=True
    )[:TOP_INPUT_FEATURE_COUNT]
    for column in fold_top_features:
        feature_selection_frequency[column] += 1

print("Feature importance averaged over 3 stratified folds.")

selected_input_columns = sorted(
    feature_importance,
    key=feature_importance.get,
    reverse=True,
)[:TOP_INPUT_FEATURE_COUNT]
# Keep both numeric and categorical inputs represented in the selected set.
if not any(column in categorical_columns for column in selected_input_columns):
    best_categorical = max(categorical_columns, key=feature_importance.get)
    selected_input_columns[-1] = best_categorical
if not any(column in all_numeric_columns for column in selected_input_columns):
    best_numeric = max(all_numeric_columns, key=feature_importance.get)
    selected_input_columns[-1] = best_numeric
selected_input_columns = list(dict.fromkeys(selected_input_columns))
print(f"\nSelected top {TOP_INPUT_FEATURE_COUNT} input features:")
for column in selected_input_columns:
    print(
        column,
        "- mean importance:", round(feature_importance[column], 4),
        "- selected in folds:", feature_selection_frequency[column], "/ 3",
    )

numeric_columns = [
    column for column in selected_input_columns if column in all_numeric_columns
]
categorical_columns = [
    column for column in selected_input_columns if column in categorical_columns
]

# Refit encoding and scaling using only the selected inputs.
encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
encoder.fit(X_train_raw.loc[model_fit_indices, categorical_columns])
X_train_categories = encoder.transform(X_train_raw[categorical_columns]).toarray()
X_validation_categories = encoder.transform(
    X_validation_raw[categorical_columns]
).toarray()
category_names = encoder.get_feature_names_out(categorical_columns)

X_train_category_data = pd.DataFrame(
    X_train_categories, columns=category_names, index=X_train_raw.index
)
X_validation_category_data = pd.DataFrame(
    X_validation_categories, columns=category_names, index=X_validation_raw.index
)
X_train = pd.concat([X_train_raw[numeric_columns], X_train_category_data], axis=1)
X_validation = pd.concat(
    [X_validation_raw[numeric_columns], X_validation_category_data], axis=1
)
feature_columns = X_train.columns.tolist()

scaler = StandardScaler()
scaler.fit(X_train.loc[model_fit_indices])
X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation[feature_columns])
model_fit_positions = X_train.index.get_indexer(model_fit_indices)
threshold_positions = X_train.index.get_indexer(threshold_indices)

print("Selected raw input count:", len(selected_input_columns))
print("Encoded model feature count:", len(feature_columns))
print("Training feature shape:", X_train_scaled.shape)
print("Validation feature shape:", X_validation_scaled.shape)
print("Hyperparameter tuning: three-fold stratified CV on a 20,000-row training sample.")


#================ OBJECTIVE 6: TRAIN AND EVALUATE MODELS ================
def find_best_threshold(actual_values, probabilities):
    """Find a probability threshold that gives the best calibration F1 score."""
    best_threshold = 0.50
    best_f1 = 0

    for threshold_value in np.arange(0.20, 0.81, 0.01):
        predictions = (probabilities >= threshold_value).astype(int)
        score = f1_score(actual_values, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold_value

    return best_threshold


def tune_model(model_name, model, parameter_grid, sample_weights=None):
    """Tune one classifier on training-only folds and return its best estimator."""
    print(f"\n================ TUNING {model_name.upper()} ================")
    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    fit_arguments = {}
    if sample_weights is not None:
        fit_arguments["sample_weight"] = sample_weights
    search.fit(X_tuning, y_tuning, **fit_arguments)
    tuning_scores[model_name] = search.best_score_
    print("Best CV average precision:", round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)
    return search.best_estimator_, search.best_params_


# Use a stratified sample for tractable tuning; final candidate fits still use
# the full model-fit partition, while calibration and final validation stay separate.
tuning_rows = min(20000, len(model_fit_positions))
tuning_sample_positions, _ = train_test_split(
    np.arange(len(model_fit_positions)),
    train_size=tuning_rows,
    random_state=RANDOM_STATE,
    stratify=y_model_fit,
)
tuning_positions = model_fit_positions[tuning_sample_positions]
X_tuning = X_train_scaled[tuning_positions]
y_tuning = y_model_fit.iloc[tuning_sample_positions]
tuned_parameters = {}
tuning_scores = {}


print("\n================ TUNING LOGISTIC REGRESSION ================")
logistic_model, tuned_parameters["Logistic Regression"] = tune_model(
    "Logistic Regression",
    LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    {"C": [0.1, 1.0, 10.0]},
)
logistic_model.fit(X_train_scaled[model_fit_positions], y_model_fit)
logistic_threshold_probabilities = logistic_model.predict_proba(
    X_train_scaled[threshold_positions]
)[:, 1]
logistic_probabilities = logistic_model.predict_proba(X_validation_scaled)[:, 1]
logistic_threshold = find_best_threshold(y_threshold, logistic_threshold_probabilities)

print("\n================ TUNING K-NEAREST NEIGHBORS ================")
knn_model, tuned_parameters["K-Nearest Neighbors"] = tune_model(
    "K-Nearest Neighbors",
    KNeighborsClassifier(n_jobs=1),
    {"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]},
)
knn_model.fit(X_train_scaled[model_fit_positions], y_model_fit)
knn_threshold_probabilities = knn_model.predict_proba(
    X_train_scaled[threshold_positions]
)[:, 1]
knn_probabilities = knn_model.predict_proba(X_validation_scaled)[:, 1]
knn_threshold = find_best_threshold(y_threshold, knn_threshold_probabilities)

print("\n================ TUNING NAIVE BAYES ================")
naive_bayes_model, tuned_parameters["Naive Bayes"] = tune_model(
    "Naive Bayes",
    GaussianNB(),
    {"var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8]},
)
naive_bayes_model.fit(X_train_scaled[model_fit_positions], y_model_fit)
naive_bayes_threshold_probabilities = naive_bayes_model.predict_proba(
    X_train_scaled[threshold_positions]
)[:, 1]
naive_bayes_probabilities = naive_bayes_model.predict_proba(X_validation_scaled)[:, 1]
naive_bayes_threshold = find_best_threshold(
    y_threshold, naive_bayes_threshold_probabilities
)

print("\n================ TUNING DECISION TREE ================")
decision_tree_model, tuned_parameters["Decision Tree"] = tune_model(
    "Decision Tree",
    DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
    {"max_depth": [4, 6, 10, None], "min_samples_leaf": [20, 100]},
)
decision_tree_model.fit(X_train_scaled[model_fit_positions], y_model_fit)
tree_threshold_probabilities = decision_tree_model.predict_proba(
    X_train_scaled[threshold_positions]
)[:, 1]
tree_probabilities = decision_tree_model.predict_proba(X_validation_scaled)[:, 1]
tree_threshold = find_best_threshold(y_threshold, tree_threshold_probabilities)

print("\n================ TUNING ADABOOST ================")
balanced_tuning_weights = compute_sample_weight("balanced", y_tuning)
ada_boost_model, tuned_parameters["AdaBoost"] = tune_model(
    "AdaBoost",
    AdaBoostClassifier(random_state=RANDOM_STATE),
    [
        {"n_estimators": [50], "learning_rate": [0.1, 0.5]},
        {"n_estimators": [100], "learning_rate": [0.1, 0.5]},
        {"n_estimators": [200], "learning_rate": [0.1, 0.5]},
    ],
    sample_weights=balanced_tuning_weights,
)
ada_boost_model.fit(
    X_train_scaled[model_fit_positions], y_model_fit,
    sample_weight=compute_sample_weight("balanced", y_model_fit),
)
ada_boost_threshold_probabilities = ada_boost_model.predict_proba(
    X_train_scaled[threshold_positions]
)[:, 1]
ada_boost_probabilities = ada_boost_model.predict_proba(X_validation_scaled)[:, 1]
ada_boost_threshold = find_best_threshold(
    y_threshold, ada_boost_threshold_probabilities
)

print("\n================ TUNING GRADIENT BOOSTING ================")
gradient_boosting_model, tuned_parameters["Gradient Boosting"] = tune_model(
    "Gradient Boosting",
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    [
        {"n_estimators": [75], "learning_rate": [0.03, 0.08], "max_depth": [1, 2]},
        {"n_estimators": [150], "learning_rate": [0.03, 0.08], "max_depth": [1, 2]},
    ],
    sample_weights=balanced_tuning_weights,
)
gradient_boosting_model.fit(
    X_train_scaled[model_fit_positions], y_model_fit,
    sample_weight=compute_sample_weight("balanced", y_model_fit),
)
gradient_threshold_probabilities = gradient_boosting_model.predict_proba(
    X_train_scaled[threshold_positions]
)[:, 1]
gradient_boosting_probabilities = gradient_boosting_model.predict_proba(
    X_validation_scaled
)[:, 1]
gradient_boosting_threshold = find_best_threshold(
    y_threshold, gradient_threshold_probabilities
)

print("\n================ TUNING XGBOOST ================")
positive_count = (y_model_fit == 1).sum()
negative_count = (y_model_fit == 0).sum()
xgboost_model, tuned_parameters["XGBoost"] = tune_model(
    "XGBoost",
    XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE,
    ),
    [
        {"n_estimators": [80, 140], "max_depth": [3], "learning_rate": [0.05],
         "subsample": [0.8], "colsample_bytree": [0.8],
         "scale_pos_weight": [negative_count / positive_count]},
        {"n_estimators": [80, 140], "max_depth": [5], "learning_rate": [0.1],
         "subsample": [0.8], "colsample_bytree": [0.8],
         "scale_pos_weight": [negative_count / positive_count]},
    ],
)
xgboost_model.fit(X_train_scaled[model_fit_positions], y_model_fit)
xgboost_threshold_probabilities = xgboost_model.predict_proba(
    X_train_scaled[threshold_positions]
)[:, 1]
xgboost_probabilities = xgboost_model.predict_proba(X_validation_scaled)[:, 1]
xgboost_threshold = find_best_threshold(y_threshold, xgboost_threshold_probabilities)

trained_models = {
    "Logistic Regression": logistic_model,
    "K-Nearest Neighbors": knn_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "AdaBoost": ada_boost_model,
    "Gradient Boosting": gradient_boosting_model,
    "XGBoost": xgboost_model,
}
model_probabilities = {
    "Logistic Regression": logistic_probabilities,
    "K-Nearest Neighbors": knn_probabilities,
    "Naive Bayes": naive_bayes_probabilities,
    "Decision Tree": tree_probabilities,
    "AdaBoost": ada_boost_probabilities,
    "Gradient Boosting": gradient_boosting_probabilities,
    "XGBoost": xgboost_probabilities,
}
model_thresholds = {
    "Logistic Regression": logistic_threshold,
    "K-Nearest Neighbors": knn_threshold,
    "Naive Bayes": naive_bayes_threshold,
    "Decision Tree": tree_threshold,
    "AdaBoost": ada_boost_threshold,
    "Gradient Boosting": gradient_boosting_threshold,
    "XGBoost": xgboost_threshold,
}

model_results = []
model_predictions = {}

for model_name, probabilities in model_probabilities.items():
    threshold_value = model_thresholds[model_name]
    predictions = (probabilities >= threshold_value).astype(int)
    model_predictions[model_name] = predictions
    model_results.append({
        "Model": model_name,
        "Threshold": threshold_value,
        "Accuracy": accuracy_score(y_validation, predictions),
        "Precision": precision_score(y_validation, predictions, zero_division=0),
        "Recall": recall_score(y_validation, predictions, zero_division=0),
        "F1 Score": f1_score(y_validation, predictions, zero_division=0),
        "ROC AUC": roc_auc_score(y_validation, probabilities),
    })

    print("\n", model_name)
    print("Threshold:", round(threshold_value, 2))
    print(classification_report(y_validation, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_validation, predictions))


#================ OBJECTIVE 7: COMPARE MODEL RESULTS ================
comparison = pd.DataFrame(model_results).sort_values(
    "F1 Score", ascending=False
).reset_index(drop=True)
print("\n================ MODEL COMPARISON ================")
print(comparison.round(4).to_string(index=False))
print("\n================ CROSS-VALIDATION TUNING SUMMARY ================")
tuning_summary = pd.DataFrame([
    {
        "Model": model_name,
        "CV Average Precision": tuning_scores[model_name],
        "Best Parameters": str(tuned_parameters[model_name]),
    }
    for model_name in trained_models
]).sort_values("CV Average Precision", ascending=False)
print(tuning_summary.round(4).to_string(index=False))

plt.figure(figsize=(10, 5))


comparison_plot = comparison.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    var_name="Metric",
    value_name="Score",
)
sns.barplot(data=comparison_plot, x="Metric", y="Score", hue="Model")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
save_graph("04_model_metrics_comparison.png")

plt.figure(figsize=(8, 6))
for model_name, probabilities in model_probabilities.items():
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_validation, probabilities
    )
    auc_value = roc_auc_score(y_validation, probabilities)
    plt.plot(false_positive_rate, true_positive_rate,
             label=f"{model_name} (AUC={auc_value:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
save_graph("05_roc_curve_comparison.png")

fig, axes = plt.subplots(3, 3, figsize=(15, 13))
for axis, (model_name, predictions) in zip(axes.flatten(), model_predictions.items()):
    matrix = confusion_matrix(y_validation, predictions)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_title(model_name)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
for axis in axes.flatten()[len(model_predictions):]:
    axis.set_visible(False)
save_graph("06_confusion_matrices.png")

FIGURES_PATH.mkdir(parents=True, exist_ok=True)
tuning_summary.to_csv(FIGURES_PATH.parent / "model_tuning_summary.csv", index=False)


#================ OBJECTIVE 8: SAVE MODEL FOR THE APP ================
# Save the best model by validation F1 score for the prediction app.
best_model_result = comparison.iloc[0]
best_model_name = best_model_result["Model"]
best_model = trained_models[best_model_name]
best_model_threshold = model_thresholds[best_model_name]

# Refit the selected model on all labelled training customers before saving.
if best_model_name in ["AdaBoost", "Gradient Boosting"]:
    best_model.fit(
        X_train_scaled,
        y_train,
        sample_weight=compute_sample_weight("balanced", y_train),
    )
elif best_model_name == "XGBoost":
    best_model.set_params(
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    )
    best_model.fit(X_train_scaled, y_train)
else:
    best_model.fit(X_train_scaled, y_train)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {
        "model": best_model,
        "encoder": encoder,
        "scaler": scaler,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "selected_input_columns": selected_input_columns,
        "feature_columns": feature_columns,
        "threshold": best_model_threshold,
        "model_name": best_model_name,
        "best_parameters": tuned_parameters[best_model_name],
        "all_tuned_parameters": tuned_parameters,
        "tuning_cv_average_precision": tuning_scores,
    },
    MODEL_PATH,
)

print("\nModel saved successfully:", MODEL_PATH)
print("Saved model:", best_model_name)
print("Saved threshold:", round(best_model_threshold, 3))
