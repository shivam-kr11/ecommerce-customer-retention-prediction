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
 sessions.csv (1.05M)* ──┼──> Data Cleaning ──> Feature Engineering ──> Leakage-Free ──> Models & Threshold
 transactions.csv (500k)─┼     & Parsing        & Ratios (Master DF)      Partitioning      Optimization
 marketing_campaigns.csv ─┤
 geo_data.csv            ─┘
```

1. **Customers Profile** (Demographics, loyalty score, credit limit, tenure).
2. **Web Sessions (1,048,575 rows currently available)** (Bounce rates, dwell times, pages viewed, cart additions). The pipeline documentation previously expected 2,000,000 rows; the local CSV has exactly 1,048,575 data rows, so the full source file is needed to restore the missing records.
3. **Transaction Logs (500,000 rows)** (Order values, items per basket, discount usage, purchase recency).
4. **Marketing Signals** (Campaign channel affinities and allocated ad budgets).
5. **Geo-economic Context** (Regional average income, urban density, and internet penetration).

---

## ⚙️ Key Feature Engineering

Rather than feeding raw aggregates straight into algorithms, we engineered cross-domain behavioral signals:

- **Cart-to-Order Conversion**: $\frac{\text{Total Transactions}}{\text{Total Cart Additions} + 1}$ — reveals purchase hesitation.
- **Monetary Spend Velocity**: $\frac{\text{Total Spend}}{\text{Total Sessions} + 1}$ — distinguishes window-shoppers from high-intent buyers.
- **Spend-to-Credit Ratio**: $\frac{\text{Total Spend}}{\text{Credit Limit} + 1}$ — measures financial engagement depth.
- **Recent Activity Windows**: Session and purchase counts over the last 30, 90, and 180 days, plus recent 90-day spend share.
- **Non-Negative Recency & Tenure**: Measured against dynamic historical observation cutoffs to prevent lookahead distortion.

---

## 🛡️ Addressing Data Leakage (Engineering Integrity)

Data leakage is the easiest way to fool yourself with fake high scores in offline ML. We audited and eliminated three major risks:

1. **No Temporal Inversion**: Dates are parsed and recency/tenure are measured against the latest event date in the available data (`2024-09-26`).
2. **Customer-Level Holdout**: The labelled customers are split into 128,000 training customers and 32,000 final validation customers. A separate 20% calibration split within training selects probability thresholds; the final validation labels are not used for threshold tuning.
3. **Training-Only Preprocessing and Feature Assessment**: Imputation, outlier caps, category encoding, scaling, and three-fold feature-importance assessment use training data only. The final validation customers are used once for model evaluation.

---

## 📊 Model Benchmark & Comparison

The pipeline ranks candidate inputs using three-fold feature-importance stability on the training split, then trains all models with the same **eight raw input features**. Numeric inputs are scaled and `region` is one-hot encoded, so the model matrix contains more than eight columns. Each of the seven classifiers receives a small `GridSearchCV` search scored by average precision, using three-fold stratified CV on a 20,000-row sample from the model-fit training partition. The search settings are saved in `reports/model_tuning_summary.csv` and the selected settings in `models/retention_model.pkl`. Thresholds are calibrated on a separate training subset; the 32,000-customer validation holdout is reserved for final comparison.

The current selected inputs are `loyalty_score`, `age`, `discount_usage_rate`, `email_open_rate`, `avg_cart_additions`, `avg_review_score`, `spend_to_credit_ratio`, and `region`.

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (saved model)** | **0.48** | 51.70% | 24.86% | 68.20% | **0.3644** | **0.6154** |
| Gradient Boosting | 0.46 | 48.01% | 24.24% | 73.40% | **0.3644** | 0.6124 |
| AdaBoost | 0.40 | 44.99% | 23.69% | **76.93%** | 0.3622 | 0.6105 |
| Logistic Regression | 0.47 | 50.18% | 24.27% | 68.57% | 0.3585 | 0.5994 |
| Decision Tree | 0.45 | 50.51% | 24.30% | 67.95% | 0.3580 | 0.6014 |
| K-Nearest Neighbors | 0.20 | **56.60%** | 22.60% | 46.90% | 0.3050 | 0.5411 |
| Naive Bayes | 0.20 | 49.64% | 20.39% | 50.95% | 0.2912 | 0.5074 |

The tuned XGBoost result is close to the earlier untuned AdaBoost F1 (0.3665), but does not improve on it. Model ranking quality and recall remain modest; tuning should be treated as a measured search, not a guaranteed performance gain.

### Threshold and class balance
Churn represents about 20.3% of labelled customers. Accuracy alone is therefore not a useful selection metric. Thresholds are calibrated separately from the final holdout; use precision and recall together to choose an operational point that fits campaign capacity and cost.

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

# Install dependencies into the active virtual environment
python -m pip install -r requirements.txt
```

On Windows PowerShell, you can run the install and app commands explicitly through the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
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

## 🖥️ Interactive Streamlit App

Run the API with `python -m uvicorn app:app --reload`, then launch the interface in a second terminal with `streamlit run streamlit_app.py`. On Windows, use the `.venv\Scripts\python.exe` and `.venv\Scripts\streamlit.exe` commands so both processes use the installed project dependencies. The app includes one prediction page, a dashboard, model insights, and session prediction history.

### Customer prediction page

The **Prediction** page collects the active model's selected features plus customer name and phone. A random customer ID is generated; name and phone are stored with the input values, predicted class, probability, threshold, model name, and timestamp in `data/submissions/prediction_inputs.json`. Only the model features are sent to the API. The **Prediction History** sidebar page reads saved records, supports class, ID, and name filtering, exports filtered results as CSV or JSON, and can remove selected records after confirmation. The JSON file can contain customer contact details; it is ignored by Git, and authentication is not implemented yet, so keep this demo local.

## 🔮 What’s Next on the Roadmap

- [x] **Boosting benchmarks**: AdaBoost, Gradient Boosting, and XGBoost are included in the current comparison.
- [ ] **Further boosting experiments**: Benchmark LightGBM and CatBoost with validation-safe class-imbalance tuning.
- [ ] **Restore Complete Session Data**: Replace the local 1,048,575-row sessions file with the expected 2,000,000-row source before retraining.
- [ ] **Time-Based Holdout**: Validate on a later observation window to measure performance on future customer behavior.
- [ ] **Target Encoding**: Replacing sparse one-hot geographic vectors with compact target-encoded regional churn rates.
- [x] **Hyperparameter tuning**: Three-fold stratified `GridSearchCV` across all seven classifiers; see `reports/model_tuning_summary.csv` for selected parameters and CV scores.
- [ ] **Further tuning**: Expand the search ranges and evaluate LightGBM/CatBoost with a nested or later time-based holdout.
