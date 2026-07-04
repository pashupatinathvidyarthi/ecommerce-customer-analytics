"""
02_churn_prediction.py
------------------------
Predicts customer churn using RFM + behavioral features.
Trains Logistic Regression -> Random Forest -> XGBoost, compares them,
and explains the best model with SHAP.

IMPORTANT METHODOLOGY NOTE (mention this in your interview — it shows
maturity): churn features are computed from an OBSERVATION WINDOW that
ends at a cutoff date. The label (churned or not) is determined by
activity in the 90 days AFTER that cutoff. This time-based split avoids
leakage — if you compute "recency" up to the same date you use to define
churn, recency literally IS the label, and your model will look like it
has perfect (and useless) accuracy.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                              confusion_matrix)
from xgboost import XGBClassifier
import shap

sns.set_style("whitegrid")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------- 1. Load data ----------------
customers = pd.read_csv("data/customers.csv", parse_dates=["signup_date"])
orders = pd.read_csv("data/orders.csv", parse_dates=["order_date"])
items = pd.read_csv("data/order_items.csv")

MAX_DATE = orders["order_date"].max()
CHURN_WINDOW_DAYS = 90
CUTOFF_DATE = MAX_DATE - pd.Timedelta(days=CHURN_WINDOW_DAYS)

print(f"Data range: {orders['order_date'].min().date()} to {MAX_DATE.date()}")
print(f"Feature observation cutoff: {CUTOFF_DATE.date()}")
print(f"Churn evaluated on activity between {CUTOFF_DATE.date()} and {MAX_DATE.date()}")

# ---------------- 2. Features computed ONLY from data up to CUTOFF_DATE ----------------
hist_orders = orders[orders["order_date"] <= CUTOFF_DATE]

order_agg = hist_orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    total_spend=("order_value", "sum"),
    avg_order_value=("order_value", "mean"),
    last_order_date=("order_date", "max"),
    first_order_date=("order_date", "min"),
    discount_orders=("discount_applied", "sum"),
).reset_index()

order_agg["recency_days"] = (CUTOFF_DATE - order_agg["last_order_date"]).dt.days
order_agg["tenure_days"] = (order_agg["last_order_date"] - order_agg["first_order_date"]).dt.days
order_agg["discount_rate"] = order_agg["discount_orders"] / order_agg["total_orders"]
order_agg["order_frequency"] = order_agg["total_orders"] / (order_agg["tenure_days"] + 1) * 30

cat_diversity = (hist_orders.merge(items, on="order_id")
                  .groupby("customer_id")["category"].nunique()
                  .reset_index(name="category_diversity"))

fav_payment = (hist_orders.groupby("customer_id")["payment_method"]
               .agg(lambda x: x.value_counts().idxmax())
               .reset_index(name="preferred_payment"))

df = customers.merge(order_agg, on="customer_id", how="left")
df = df.merge(cat_diversity, on="customer_id", how="left")
df = df.merge(fav_payment, on="customer_id", how="left")

df = df[pd.to_datetime(df["signup_date"]) <= CUTOFF_DATE].copy()

df["total_orders"] = df["total_orders"].fillna(0)
df["category_diversity"] = df["category_diversity"].fillna(0)
df["recency_days"] = df["recency_days"].fillna((CUTOFF_DATE - pd.to_datetime(df["signup_date"])).dt.days)
df["order_frequency"] = df["order_frequency"].fillna(0)
df["discount_rate"] = df["discount_rate"].fillna(0)
df["avg_order_value"] = df["avg_order_value"].fillna(0)
df["total_spend"] = df["total_spend"].fillna(0)
df["preferred_payment"] = df["preferred_payment"].fillna("None")

# ---------------- 3. Label computed from FUTURE window (CUTOFF -> MAX_DATE) ----------------
future_orders = orders[(orders["order_date"] > CUTOFF_DATE) & (orders["order_date"] <= MAX_DATE)]
active_future_customers = set(future_orders["customer_id"].unique())
df["churned"] = (~df["customer_id"].isin(active_future_customers)).astype(int)

print("\nChurn rate (next 90 days, out-of-sample):", round(df["churned"].mean(), 3))

# ---------------- 4. Prepare model matrix ----------------
feature_cols = ["total_orders", "total_spend", "avg_order_value", "recency_days",
                 "order_frequency", "discount_rate", "category_diversity", "age"]
cat_cols = ["gender", "acquisition_channel", "preferred_payment", "city"]

model_df = df[feature_cols + cat_cols + ["churned"]].copy()
model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

X = model_df.drop(columns=["churned"])
y = model_df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- 5. Train models ----------------
results = {}

lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train_scaled, y_train)
results["Logistic Regression"] = lr.predict_proba(X_test_scaled)[:, 1]

rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
results["Random Forest"] = rf.predict_proba(X_test)[:, 1]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                     scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
results["XGBoost"] = xgb.predict_proba(X_test)[:, 1]

# ---------------- 6. Evaluate & compare ----------------
print("\n===== MODEL COMPARISON =====")
metrics_summary = []
for name, probs in results.items():
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)
    metrics_summary.append({
        "model": name, "roc_auc": round(auc, 3),
        "precision": round(report["1"]["precision"], 3),
        "recall": round(report["1"]["recall"], 3),
        "f1": round(report["1"]["f1-score"], 3),
    })
    print(f"\n--- {name} ---  ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, preds))

metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv(f"{OUT}/churn_model_comparison.csv", index=False)

# ---------------- 7. Plots ----------------
plt.figure(figsize=(7, 6))
for name, probs in results.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Churn Prediction Model Comparison")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/churn_roc_comparison.png", dpi=150); plt.close()

best_model_name = metrics_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
best_probs = results[best_model_name]
best_preds = (best_probs >= 0.5).astype(int)
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/churn_confusion_matrix.png", dpi=150); plt.close()

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

plt.figure()
shap.summary_plot(shap_values, X_test, show=False, max_display=12)
plt.title("SHAP Feature Importance (XGBoost)")
plt.tight_layout()
plt.savefig(f"{OUT}/churn_shap_importance.png", dpi=150, bbox_inches="tight"); plt.close()

print(f"\nBest model by ROC-AUC: {best_model_name}")
print("Saved: churn_model_comparison.csv, churn_roc_comparison.png, churn_confusion_matrix.png, churn_shap_importance.png")

df.to_csv(f"{OUT}/customer_features_full.csv", index=False)
print("Saved feature dataset ->", f"{OUT}/customer_features_full.csv")
