# 🛒 E-Commerce Customer Retention & Churn Intelligence

Predicting customer churn before it happens is one of the highest-ROI challenges in modern e-commerce. Acquiring a new shopper costs roughly 5x more than keeping an existing one—yet most stores only realize a customer is gone when they stop coming back.

This project builds an end-to-end, **leakage-free machine learning pipeline** that blends behavioral clickstream data, transaction histories, marketing exposure, and geo-demographics across **200,000 customers** to identify who is likely to churn and when to step in.

---

## 🎯 The Core Business Problem

- **Objective**: Predict whether an e-commerce customer will churn (`target = 1`) or remain active/retained (`target = 0`).
- **Class Imbalance**: ~79.7% of customers are retained; ~20.3% churn. A naive model predicting "everyone stays" would score 80% accuracy while failing 100% of the business goal.
- **Goal**: Maximize **Recall** (catching churners early) and **ROC-AUC** while maintaining a viable **Precision** so retention campaigns aren't wasted.

---

## 🏗️ Architecture & Data Ecosystem

The intelligence engine merges five distinct operational sources into a unified feature store:

```
 customers.csv (200k)  ──┐
 sessions.csv (2M)     ──┼──> Data Cleaning ──> Feature Engineering ──> Leakage-Free ──> Models & Threshold
 transactions.csv (500k)─┼     & Parsing        & Ratios (Master DF)      Partitioning      Optimization
 marketing_campaigns.csv ─┤
 geo_data.csv            ─┘
```

1. **Customers Profile** (Demographics, loyalty score, credit limit, tenure).
2. **Web Sessions (2,000,000 rows)** (Bounce rates, dwell times, pages viewed, cart additions).
3. **Transaction Logs (500,000 rows)** (Order values, items per basket, discount usage, purchase recency).
4. **Marketing Signals** (Campaign channel affinities and allocated ad budgets).
5. **Geo-economic Context** (Regional average income, urban density, and internet penetration).

---

## ⚙️ Key Feature Engineering

Rather than feeding raw aggregates straight into algorithms, we engineered cross-domain behavioral signals:

- **Cart-to-Order Conversion**: $\frac{\text{Total Transactions}}{\text{Total Cart Additions} + 1}$ — reveals purchase hesitation.
- **Monetary Spend Velocity**: $\frac{\text{Total Spend}}{\text{Total Sessions} + 1}$ — distinguishes window-shoppers from high-intent buyers.
- **Spend-to-Credit Ratio**: $\frac{\text{Total Spend}}{\text{Credit Limit} + 1}$ — measures financial engagement depth.
- **Non-Negative Recency & Tenure**: Measured against dynamic historical observation cutoffs to prevent lookahead distortion.

---

## 🛡️ Addressing Data Leakage (Engineering Integrity)

Data leakage is the easiest way to fool yourself with fake high scores in offline ML. We audited and eliminated three major risks:

1. **No Temporal Inversion**: Replaced static reference dates with dynamic observation boundaries (`2024-09-26`), completely eliminating negative tenures and negative recency values.
2. **Upfront Customer-Level Partitioning**: All median imputations (`age`, `email_open_rate`, `credit_limit`, `pages_viewed`, `average_income`) and 99th percentile outlier clipping thresholds are calculated **strictly on training customers** (`128,000` samples) before touching the validation set (`32,000` samples).
3. **Isolated Preprocessing**: `OneHotEncoder(handle_unknown="ignore")` and `StandardScaler` are fit exclusively on `X_train` and applied downstream to `X_val`.

---

## 📊 Model Benchmark & Comparison

All models were evaluated on an identical **32,000-customer stratified validation holdout**:

| Model | Decision Threshold ($p$) | Accuracy | Precision | Recall (Churn Caught) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🌲 **Decision Tree (Tuned)** | **$p = 0.47$** | **51.70%** | **24.67%** | **67.15%** | **0.3609** | **0.6040** |
| 🌲 Decision Tree (Default) | $p = 0.50$ | 56.17% | 25.28% | 59.26% | 0.3544 | **0.6040** |
| 📈 **Logistic Regression (Tuned)**| **$p = 0.46$** | **47.98%** | **23.90%** | **71.53%** | **0.3583** | **0.5987** |
| 📈 Logistic Regression (Default)| $p = 0.50$ | 57.93% | 25.35% | 55.12% | 0.3472 | 0.5987 |
| 🔍 K-Nearest Neighbors | $p = 0.50$ | 76.12% | 20.65% | 6.20% | 0.0954 | 0.5018 |
| 🎲 Naive Bayes (Gaussian) | $p = 0.50$ | 21.00% | 20.32% | 98.95% | 0.3371 | 0.5123 |

### 💡 Why Tuning the Threshold Matters
Because churners make up only ~20% of the customer base, defaulting to standard $p = 0.50$ leaves money on the table.
- Lowering the Decision Tree cutoff to **$p = 0.47$** boosted recall from **59.26% to 67.15%** (**+513 additional churners caught**).
- Lowering the Logistic Regression cutoff to **$p = 0.46$** pushed recall to **71.53%** (**+1,066 churners caught**).
- In real-world e-commerce, offering a coupon to a slightly borderline shopper is vastly cheaper than permanently losing a customer.

---

## 📂 Project Structure

```text
ecommerce-retention-intelligence/
├── data/
│   └── raw/                       # Raw CSVs (customers, sessions, transactions, etc.)
├── main.py                        # Complete pipeline: EDA, feature engineering, models, plots
├── inspect_data.py                # Initial exploratory data profiling & diagnostics
├── README.md                      # Project documentation and performance review
└── .gitignore                     # Data and cache exclusion rules
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/shivam-kr11/ecommerce-customer-retention-prediction.git
cd ecommerce-retention-intelligence

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 2. Add Raw Data
Ensure your raw CSV files are placed in `data/raw/`:
- `customers.csv`, `sessions.csv`, `transactions.csv`, `marketing_campaigns.csv`, `geo_data.csv`, `train.csv`, `test.csv`

### 3. Run the Pipeline
```bash
python main.py
```
This executes the data cleaning, feature creation, leak-free validation split, model training, threshold evaluation, and generates comparison visual charts.

---

## 🔮 What’s Next on the Roadmap

- [ ] **Gradient Boosted Ensembles**: Benchmarking LightGBM, XGBoost, and CatBoost with `scale_pos_weight` tuning.
- [ ] **RFM Velocity Features**: Adding 30d vs. 90d activity decay momentum features.
- [ ] **Target Encoding**: Replacing sparse one-hot geographic vectors with compact target-encoded regional churn rates.
- [ ] **Automated Hyperparameter Tuning**: Integrating Optuna with 5-fold Stratified Cross Validation.
