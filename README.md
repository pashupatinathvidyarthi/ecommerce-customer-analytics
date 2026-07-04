# E-Commerce Sales & Customer Insights — Extended ML Case Study

An end-to-end analytics project covering **SQL-style aggregation, Python/Pandas feature engineering, machine learning, and statistical experimentation** — built to demonstrate the skill set an Analyst / Associate Data Scientist role expects.

## Project Structure

```
ecommerce_ml_case_study/
├── data/
│   ├── customers.csv          # 5,000 synthetic customers
│   ├── orders.csv              # ~35,000 orders (Jan 2024 - Jun 2026)
│   └── order_items.csv         # ~88,000 line items
├── outputs/                    # all generated plots, tables, model results
├── 01_generate_data.py         # synthetic dataset generator
├── 02_churn_prediction.py      # churn model (LR -> RF -> XGBoost) + SHAP
├── 03_clv_modeling.py          # predictive CLV regression + cohort analysis
├── 04_ab_test_simulation.py    # power analysis + A/B test + business recommendation
└── README.md
```

> **Note on the dataset**: this is a synthetic dataset generated to mimic realistic e-commerce behavior (varying customer segments, seasonality-free but volume-realistic order patterns). If you have your own e-commerce dataset (e.g. from Kaggle), just replace the files in `data/` with matching column names — `customer_id`, `order_id`, `order_date`, `order_value` etc. — and every downstream script will still run.

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn shap faker

python 01_generate_data.py         # generates data/*.csv
python 02_churn_prediction.py      # churn model + plots
python 03_clv_modeling.py          # CLV regression + cohort analysis
python 04_ab_test_simulation.py    # A/B test simulation
```

Each script is independent after data generation and prints results to the console plus saves plots/tables to `outputs/`.

## Interactive Streamlit Dashboard

All three pieces of this project are also available as one interactive dashboard — much better for showing an interviewer live, since you can change assumptions and watch the models/results update in real time.

```bash
pip install streamlit plotly

streamlit run app.py
```

This opens a browser dashboard with four pages (navigate via the sidebar):

- **Overview** — headline metrics, revenue by category, monthly trend, acquisition channel and payment method breakdowns.
- **Churn Prediction** — drag a slider to change the churn window (30–180 days) and watch the churn rate and all three models retrain live, with updated ROC curves and feature importance.
- **CLV & Cohort Analysis** — predictive CLV scatter plot + feature importance, plus the interactive cohort retention heatmap and cumulative revenue curves.
- **A/B Test Simulator** — set your own baseline conversion rate, minimum detectable effect, alpha, and power; click "Run Simulation" to see the power analysis, simulated experiment results, and an auto-generated business recommendation.


---

## 1. Customer Churn Prediction

**Business question**: Which customers are likely to stop purchasing in the next 90 days, and why?

**Methodology (important — mention this in interviews)**: Features are computed from an *observation window* ending at a cutoff date; the churn label is determined by activity in the *following* 90 days. This time-based split prevents leakage — an earlier version of this pipeline that computed "recency" and the churn label from the same snapshot date scored a suspicious 1.00 AUC, which is a classic sign of leakage, not a good model.

**Models compared**: Logistic Regression (interpretable baseline) → Random Forest → XGBoost, all with class-weighting to handle the imbalanced churn rate (~72% in this synthetic data, class imbalance handled via `class_weight="balanced"` / `scale_pos_weight`).

**Results** (see `outputs/churn_model_comparison.csv`):
| Model | ROC-AUC | Precision (churn) | Recall (churn) | F1 |
|---|---|---|---|---|
| Logistic Regression | ~0.95 | ~0.97 | ~0.84 | ~0.90 |
| Random Forest | ~0.95 | ~0.97 | ~0.85 | ~0.90 |
| XGBoost | ~0.95 | ~0.96 | ~0.85 | ~0.90 |

**Key drivers of churn** (via SHAP, see `outputs/churn_shap_importance.png`): recency of last order, order frequency, total historical spend, and discount usage rate are the top features — consistent with standard RFM theory.


---

## 2. Customer Lifetime Value (CLV)

**Two approaches, deliberately chosen to show breadth:**

### A) Predictive CLV (regression)
Predicts a customer's spend in the *next 90 days* from historical RFM features. Compared Linear Regression (baseline) vs Gradient Boosting.

**Results** (`outputs/clv_model_comparison.csv`):
- Linear Regression: R² ≈ 0.42
- Gradient Boosting: R² ≈ 0.62, lower MAE

The gap between the two models is itself a useful data story — it says customer future spend has meaningful non-linear structure (e.g., diminishing returns on frequency, interaction effects between recency and spend) that a linear model can't capture.

### B) Cohort Analysis
Groups customers by signup month, tracks retention (`outputs/cohort_retention_heatmap.png`) and cumulative average revenue per customer (`outputs/cohort_clv_curves.png`) over their lifetime. This is the model-free approach most business stakeholders actually prefer for CLV — it's a table/chart they can read without trusting a black box.


---

## 3. A/B Test Simulation

**Business question**: "If we offer a 10% discount at checkout, does it increase conversion enough to justify the margin hit?"

**Full workflow**:
1. **Power analysis** — calculated the required sample size (≈5,438 per group) to detect a 15% relative lift in conversion rate at α=0.05, power=0.80, using Cohen's h effect size for proportions.
2. **Simulated the experiment** with the designed effect size to validate the test setup.
3. **Two-proportion z-test** on conversion rate → statistically significant lift (p ≈ 0.008).
4. **Independent t-test** on average order value → also significant, but in the *opposite* direction (discount reduces AOV).
5. **Business recommendation**: despite the AOV drop, revenue-per-visitor still improved (₹212 → ₹217), so the net recommendation is to roll out — with a note to re-test smaller discount tiers to find the margin-optimal point, and watch for novelty effects over the following weeks.

This is the piece that most directly matches the JD's explicit callout of "Hypothesis testing, Sample size estimation, A/B testing."

---

## Files Generated (in `outputs/`)

| File | Description |
|---|---|
| `churn_model_comparison.csv` | Precision/recall/F1/AUC for all 3 churn models |
| `churn_roc_comparison.png` | ROC curves overlay |
| `churn_confusion_matrix.png` | Confusion matrix for best model |
| `churn_shap_importance.png` | SHAP feature importance / direction |
| `customer_features_full.csv` | Full feature-engineered dataset |
| `clv_model_comparison.csv` | MAE/R² for CLV regression models |
| `clv_actual_vs_predicted.png` | Scatter plot of predicted vs actual spend |
| `clv_feature_importance.png` | Top drivers of predicted CLV |
| `cohort_retention_heatmap.png` | Retention % by signup cohort over time |
| `cohort_clv_curves.png` | Cumulative revenue per customer by cohort |
| `cohort_retention_matrix.csv` / `cohort_avg_clv_matrix.csv` | Underlying data for the above |
| `ab_test_power_curve.png` | Power vs sample size curve |
| `ab_test_results.png` | Conversion rate + AOV comparison with error bars |
| `ab_test_summary.csv` | All key numbers from the experiment |
