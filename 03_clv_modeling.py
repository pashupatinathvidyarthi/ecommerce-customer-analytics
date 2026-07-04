"""
03_clv_modeling.py
--------------------
Two complementary approaches to Customer Lifetime Value:

  A) Predictive CLV: regression model predicting a customer's spend in
     the NEXT 90 days, based on their historical RFM behavior (same
     time-based cutoff logic as the churn model, to avoid leakage).

  B) Cohort analysis: group customers by signup month, track retention
     and cumulative revenue per cohort over time — the classic way
     analytics teams visualize CLV trends without needing a model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_style("whitegrid")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

customers = pd.read_csv("data/customers.csv", parse_dates=["signup_date"])
orders = pd.read_csv("data/orders.csv", parse_dates=["order_date"])

MAX_DATE = orders["order_date"].max()
WINDOW_DAYS = 90
CUTOFF_DATE = MAX_DATE - pd.Timedelta(days=WINDOW_DAYS)

# =========================================================
# PART A — Predictive CLV (next-90-day spend regression)
# =========================================================
hist = orders[orders["order_date"] <= CUTOFF_DATE]
future = orders[(orders["order_date"] > CUTOFF_DATE) & (orders["order_date"] <= MAX_DATE)]

hist_agg = hist.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    total_spend=("order_value", "sum"),
    avg_order_value=("order_value", "mean"),
    last_order_date=("order_date", "max"),
    first_order_date=("order_date", "min"),
).reset_index()
hist_agg["recency_days"] = (CUTOFF_DATE - hist_agg["last_order_date"]).dt.days
hist_agg["tenure_days"] = (hist_agg["last_order_date"] - hist_agg["first_order_date"]).dt.days
hist_agg["order_frequency"] = hist_agg["total_orders"] / (hist_agg["tenure_days"] + 1) * 30

future_spend = future.groupby("customer_id")["order_value"].sum().reset_index(name="future_90d_spend")

clv_df = customers.merge(hist_agg, on="customer_id", how="left")
clv_df = clv_df[pd.to_datetime(clv_df["signup_date"]) <= CUTOFF_DATE].copy()
clv_df = clv_df.merge(future_spend, on="customer_id", how="left")
clv_df["future_90d_spend"] = clv_df["future_90d_spend"].fillna(0)

for c in ["total_orders", "total_spend", "avg_order_value", "recency_days", "order_frequency"]:
    clv_df[c] = clv_df[c].fillna(0 if c != "recency_days" else (CUTOFF_DATE - pd.to_datetime(clv_df["signup_date"])).dt.days)

feature_cols = ["total_orders", "total_spend", "avg_order_value", "recency_days", "order_frequency", "age"]
cat_cols = ["gender", "acquisition_channel", "city"]

model_df = clv_df[feature_cols + cat_cols + ["future_90d_spend"]].copy()
model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

X = model_df.drop(columns=["future_90d_spend"])
y = model_df["future_90d_spend"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lin = LinearRegression()
lin.fit(X_train, y_train)
lin_preds = lin.predict(X_test)

gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
gbr.fit(X_train, y_train)
gbr_preds = gbr.predict(X_test)

print("===== CLV REGRESSION RESULTS (predicting next-90-day spend) =====")
for name, preds in [("Linear Regression", lin_preds), ("Gradient Boosting", gbr_preds)]:
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE = Rs.{mae:,.0f}, R2 = {r2:.3f}")

# Actual vs predicted plot (best model)
best_preds = gbr_preds if r2_score(y_test, gbr_preds) > r2_score(y_test, lin_preds) else lin_preds
best_name = "Gradient Boosting" if best_preds is gbr_preds else "Linear Regression"

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.3, s=15)
max_val = max(y_test.max(), best_preds.max())
plt.plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
plt.xlabel("Actual 90-day spend (Rs.)")
plt.ylabel("Predicted 90-day spend (Rs.)")
plt.title(f"CLV Prediction: Actual vs Predicted ({best_name})")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/clv_actual_vs_predicted.png", dpi=150); plt.close()

# Feature importance (if GBR is best)
if best_name == "Gradient Boosting":
    importances = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Features Driving Predicted CLV")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT}/clv_feature_importance.png", dpi=150); plt.close()

clv_summary = pd.DataFrame([
    {"model": "Linear Regression", "mae": round(mean_absolute_error(y_test, lin_preds), 2), "r2": round(r2_score(y_test, lin_preds), 3)},
    {"model": "Gradient Boosting", "mae": round(mean_absolute_error(y_test, gbr_preds), 2), "r2": round(r2_score(y_test, gbr_preds), 3)},
])
clv_summary.to_csv(f"{OUT}/clv_model_comparison.csv", index=False)

# =========================================================
# PART B — Cohort Analysis
# =========================================================
orders_full = orders.merge(customers[["customer_id", "signup_date"]], on="customer_id")
orders_full["cohort_month"] = pd.to_datetime(orders_full["signup_date"]).dt.to_period("M")
orders_full["order_month"] = orders_full["order_date"].dt.to_period("M")
orders_full["cohort_index"] = (orders_full["order_month"] - orders_full["cohort_month"]).apply(lambda x: x.n)

# Retention: unique customers active in each cohort_index, relative to cohort size
cohort_data = orders_full.groupby(["cohort_month", "cohort_index"])["customer_id"].nunique().reset_index()
cohort_sizes = orders_full.groupby("cohort_month")["customer_id"].nunique()
cohort_pivot = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="customer_id")
retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)

# Revenue per cohort per month (cumulative)
revenue_data = orders_full.groupby(["cohort_month", "cohort_index"])["order_value"].sum().reset_index()
revenue_pivot = revenue_data.pivot(index="cohort_month", columns="cohort_index", values="order_value")
cumulative_revenue = revenue_pivot.cumsum(axis=1)
avg_cumulative_clv = cumulative_revenue.divide(cohort_sizes, axis=0)

# Plot retention heatmap (limit to first 12 months, first 15 cohorts for readability)
plt.figure(figsize=(12, 7))
sns.heatmap(retention_matrix.iloc[:15, :13], annot=True, fmt=".0%", cmap="YlGnBu", cbar_kws={"label": "Retention %"})
plt.title("Cohort Retention Heatmap (by signup month)")
plt.xlabel("Months Since Signup")
plt.ylabel("Signup Cohort")
plt.tight_layout()
plt.savefig(f"{OUT}/cohort_retention_heatmap.png", dpi=150); plt.close()

# Plot average cumulative CLV curves for a few cohorts
plt.figure(figsize=(9, 6))
for cohort in avg_cumulative_clv.index[:8]:
    plt.plot(avg_cumulative_clv.columns, avg_cumulative_clv.loc[cohort], marker="o", label=str(cohort))
plt.xlabel("Months Since Signup")
plt.ylabel("Avg Cumulative Revenue per Customer (Rs.)")
plt.title("Cumulative CLV Curves by Signup Cohort")
plt.legend(title="Cohort", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"{OUT}/cohort_clv_curves.png", dpi=150); plt.close()

retention_matrix.to_csv(f"{OUT}/cohort_retention_matrix.csv")
avg_cumulative_clv.to_csv(f"{OUT}/cohort_avg_clv_matrix.csv")

print("\nSaved: clv_model_comparison.csv, clv_actual_vs_predicted.png, clv_feature_importance.png")
print("Saved: cohort_retention_heatmap.png, cohort_clv_curves.png, cohort_retention_matrix.csv, cohort_avg_clv_matrix.csv")
