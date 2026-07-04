"""
app.py
-------
Interactive Streamlit dashboard for the E-Commerce ML Case Study.

Brings together all three pieces of the project in one place:
  1. Customer Churn Prediction (with an adjustable churn-window slider)
  2. CLV & Cohort Analysis
  3. A/B Test Simulator (fully interactive — change assumptions, re-run live)

Run with:
    pip install streamlit plotly pandas numpy scikit-learn xgboost statsmodels shap
    streamlit run app.py

Make sure you've already run 01_generate_data.py at least once so that
the data/ folder exists.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                              confusion_matrix, mean_absolute_error, r2_score)
from xgboost import XGBClassifier
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.weightstats import ttest_ind

st.set_page_config(page_title="E-Commerce Analytics Dashboard", page_icon="📊", layout="wide")

DATA_DIR = "data"

# ============================================================
# DATA LOADING (cached so it only runs once per session)
# ============================================================
@st.cache_data
def load_data():
    if not os.path.exists(f"{DATA_DIR}/customers.csv"):
        return None, None, None
    customers = pd.read_csv(f"{DATA_DIR}/customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv", parse_dates=["order_date"])
    items = pd.read_csv(f"{DATA_DIR}/order_items.csv")
    return customers, orders, items


customers, orders, items = load_data()

if customers is None:
    st.error(
        "No data found. Please run `python 01_generate_data.py` first "
        "from the project folder, then restart this app."
    )
    st.stop()

MAX_DATE = orders["order_date"].max()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📊 E-Commerce Analytics")
page = st.sidebar.radio(
    "Go to:",
    ["Overview", "Churn Prediction", "CLV & Cohort Analysis", "A/B Test Simulator"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(customers):,} customers, {len(orders):,} orders")
st.sidebar.caption(f"Date range: {orders['order_date'].min().date()} to {MAX_DATE.date()}")

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    st.title("E-Commerce Sales & Customer Insights")
    st.caption("An end-to-end analytics case study: churn prediction, CLV modeling, and A/B testing")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(customers):,}")
    col2.metric("Total Orders", f"{len(orders):,}")
    col3.metric("Total Revenue", f"₹{orders['order_value'].sum():,.0f}")
    col4.metric("Avg Order Value", f"₹{orders['order_value'].mean():,.0f}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by Category")
        cat_rev = items.merge(orders[["order_id"]], on="order_id").copy()
        cat_rev["revenue"] = items["item_price"] * items["quantity"]
        cat_summary = cat_rev.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(cat_summary, x="revenue", y="category", orientation="h",
                     labels={"revenue": "Revenue (₹)", "category": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Monthly Revenue Trend")
        monthly = orders.copy()
        monthly["month"] = monthly["order_date"].dt.to_period("M").astype(str)
        monthly_rev = monthly.groupby("month")["order_value"].sum().reset_index()
        fig = px.line(monthly_rev, x="month", y="order_value", markers=True,
                      labels={"order_value": "Revenue (₹)", "month": ""})
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Customers by Acquisition Channel")
        chan = customers["acquisition_channel"].value_counts().reset_index()
        chan.columns = ["channel", "count"]
        fig = px.pie(chan, values="count", names="channel", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Payment Method Distribution")
        pay = orders["payment_method"].value_counts().reset_index()
        pay.columns = ["method", "count"]
        fig = px.bar(pay, x="method", y="count", labels={"count": "Orders", "method": ""})
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Use the sidebar to explore **Churn Prediction** (adjust the churn window and see models "
        "retrain live), **CLV & Cohort Analysis**, or run your own **A/B Test Simulation**."
    )

# ============================================================
# PAGE 2 — CHURN PREDICTION
# ============================================================
elif page == "Churn Prediction":
    st.title("Customer Churn Prediction")
    st.caption(
        "Features are computed up to a cutoff date; churn is defined by inactivity in the "
        "window AFTER that cutoff — a time-based split that avoids data leakage."
    )

    churn_window = st.slider("Churn window (days of inactivity to count as churned)", 30, 180, 90, step=15)

    @st.cache_data(show_spinner="Engineering features and training models...")
    def run_churn_pipeline(window_days):
        cutoff = MAX_DATE - pd.Timedelta(days=window_days)
        hist = orders[orders["order_date"] <= cutoff]
        future = orders[(orders["order_date"] > cutoff) & (orders["order_date"] <= MAX_DATE)]

        agg = hist.groupby("customer_id").agg(
            total_orders=("order_id", "count"),
            total_spend=("order_value", "sum"),
            avg_order_value=("order_value", "mean"),
            last_order_date=("order_date", "max"),
            first_order_date=("order_date", "min"),
            discount_orders=("discount_applied", "sum"),
        ).reset_index()
        agg["recency_days"] = (cutoff - agg["last_order_date"]).dt.days
        agg["tenure_days"] = (agg["last_order_date"] - agg["first_order_date"]).dt.days
        agg["discount_rate"] = agg["discount_orders"] / agg["total_orders"]
        agg["order_frequency"] = agg["total_orders"] / (agg["tenure_days"] + 1) * 30

        cat_div = (hist.merge(items, on="order_id").groupby("customer_id")["category"]
                   .nunique().reset_index(name="category_diversity"))

        df = customers.merge(agg, on="customer_id", how="left")
        df = df.merge(cat_div, on="customer_id", how="left")
        df = df[pd.to_datetime(df["signup_date"]) <= cutoff].copy()

        df["total_orders"] = df["total_orders"].fillna(0)
        df["category_diversity"] = df["category_diversity"].fillna(0)
        df["recency_days"] = df["recency_days"].fillna((cutoff - pd.to_datetime(df["signup_date"])).dt.days)
        df["order_frequency"] = df["order_frequency"].fillna(0)
        df["discount_rate"] = df["discount_rate"].fillna(0)
        df["avg_order_value"] = df["avg_order_value"].fillna(0)
        df["total_spend"] = df["total_spend"].fillna(0)

        active_future = set(future["customer_id"].unique())
        df["churned"] = (~df["customer_id"].isin(active_future)).astype(int)

        feature_cols = ["total_orders", "total_spend", "avg_order_value", "recency_days",
                         "order_frequency", "discount_rate", "category_diversity", "age"]
        cat_cols = ["gender", "acquisition_channel"]
        model_df = df[feature_cols + cat_cols + ["churned"]].copy()
        model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

        X = model_df.drop(columns=["churned"])
        y = model_df["churned"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X_train_s, y_train)
        lr_probs = lr.predict_proba(X_test_s)[:, 1]

        rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]

        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08,
                             scale_pos_weight=spw, eval_metric="logloss", random_state=42)
        xgb.fit(X_train, y_train)
        xgb_probs = xgb.predict_proba(X_test)[:, 1]

        return {
            "churn_rate": y.mean(),
            "y_test": y_test,
            "models": {"Logistic Regression": lr_probs, "Random Forest": rf_probs, "XGBoost": xgb_probs},
            "rf_model": rf, "feature_names": X.columns.tolist(),
        }

    result = run_churn_pipeline(churn_window)

    st.metric("Churn Rate (next window, out-of-sample)", f"{result['churn_rate']:.1%}")

    # Model comparison table
    rows = []
    for name, probs in result["models"].items():
        preds = (probs >= 0.5).astype(int)
        report = classification_report(result["y_test"], preds, output_dict=True)
        rows.append({
            "Model": name,
            "ROC-AUC": round(roc_auc_score(result["y_test"], probs), 3),
            "Precision": round(report["1"]["precision"], 3),
            "Recall": round(report["1"]["recall"], 3),
            "F1": round(report["1"]["f1-score"], 3),
        })
    st.subheader("Model Comparison")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC Curves")
        fig = go.Figure()
        for name, probs in result["models"].items():
            fpr, tpr, _ = roc_curve(result["y_test"], probs)
            auc = roc_auc_score(result["y_test"], probs)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Feature Importance (Random Forest)")
        importances = pd.Series(result["rf_model"].feature_importances_, index=result["feature_names"])
        importances = importances.sort_values(ascending=True).tail(10)
        fig = px.bar(x=importances.values, y=importances.index, orientation="h",
                     labels={"x": "Importance", "y": ""})
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Tip: drag the slider above to a very short window (e.g. 30 days) vs a long one (e.g. 180) "
        "and watch how the churn rate and model performance shift — a good talking point about "
        "how the business definition of 'churn' changes the whole analysis."
    )

# ============================================================
# PAGE 3 — CLV & COHORT ANALYSIS
# ============================================================
elif page == "CLV & Cohort Analysis":
    st.title("Customer Lifetime Value & Cohort Analysis")

    tab1, tab2 = st.tabs(["Predictive CLV", "Cohort Retention"])

    with tab1:
        st.caption("Predicting each customer's spend in the next 90 days from their historical behavior.")

        @st.cache_data(show_spinner="Training CLV models...")
        def run_clv_pipeline():
            cutoff = MAX_DATE - pd.Timedelta(days=90)
            hist = orders[orders["order_date"] <= cutoff]
            future = orders[(orders["order_date"] > cutoff) & (orders["order_date"] <= MAX_DATE)]

            agg = hist.groupby("customer_id").agg(
                total_orders=("order_id", "count"), total_spend=("order_value", "sum"),
                avg_order_value=("order_value", "mean"), last_order_date=("order_date", "max"),
                first_order_date=("order_date", "min"),
            ).reset_index()
            agg["recency_days"] = (cutoff - agg["last_order_date"]).dt.days
            agg["tenure_days"] = (agg["last_order_date"] - agg["first_order_date"]).dt.days
            agg["order_frequency"] = agg["total_orders"] / (agg["tenure_days"] + 1) * 30

            future_spend = future.groupby("customer_id")["order_value"].sum().reset_index(name="future_spend")

            df = customers.merge(agg, on="customer_id", how="left")
            df = df[pd.to_datetime(df["signup_date"]) <= cutoff].copy()
            df = df.merge(future_spend, on="customer_id", how="left")
            df["future_spend"] = df["future_spend"].fillna(0)
            for c in ["total_orders", "total_spend", "avg_order_value", "order_frequency"]:
                df[c] = df[c].fillna(0)
            df["recency_days"] = df["recency_days"].fillna((cutoff - pd.to_datetime(df["signup_date"])).dt.days)

            feature_cols = ["total_orders", "total_spend", "avg_order_value", "recency_days", "order_frequency", "age"]
            cat_cols = ["gender", "acquisition_channel"]
            model_df = df[feature_cols + cat_cols + ["future_spend"]].copy()
            model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

            X = model_df.drop(columns=["future_spend"])
            y = model_df["future_spend"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            lin = LinearRegression().fit(X_train, y_train)
            gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42).fit(X_train, y_train)

            return {
                "y_test": y_test, "lin_preds": lin.predict(X_test), "gbr_preds": gbr.predict(X_test),
                "gbr_model": gbr, "feature_names": X.columns.tolist(),
            }

        clv = run_clv_pipeline()
        c1, c2 = st.columns(2)
        c1.metric("Linear Regression R²", f"{r2_score(clv['y_test'], clv['lin_preds']):.3f}")
        c2.metric("Gradient Boosting R²", f"{r2_score(clv['y_test'], clv['gbr_preds']):.3f}")

        fig = px.scatter(x=clv["y_test"], y=clv["gbr_preds"], opacity=0.4,
                          labels={"x": "Actual 90-day spend (₹)", "y": "Predicted 90-day spend (₹)"},
                          title="Predicted vs Actual CLV (Gradient Boosting)")
        max_val = max(clv["y_test"].max(), clv["gbr_preds"].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                  line=dict(dash="dash", color="red"), name="Perfect prediction"))
        st.plotly_chart(fig, use_container_width=True)

        importances = pd.Series(clv["gbr_model"].feature_importances_, index=clv["feature_names"]).sort_values(ascending=True).tail(8)
        fig2 = px.bar(x=importances.values, y=importances.index, orientation="h",
                      labels={"x": "Importance", "y": ""}, title="Top Drivers of Predicted CLV")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.caption("Retention and cumulative revenue tracked by monthly signup cohort.")

        @st.cache_data(show_spinner="Building cohort tables...")
        def build_cohorts():
            of = orders.merge(customers[["customer_id", "signup_date"]], on="customer_id")
            of["cohort_month"] = pd.to_datetime(of["signup_date"]).dt.to_period("M")
            of["order_month"] = of["order_date"].dt.to_period("M")
            of["cohort_index"] = (of["order_month"] - of["cohort_month"]).apply(lambda x: x.n)

            cohort_sizes = of.groupby("cohort_month")["customer_id"].nunique()
            cohort_data = of.groupby(["cohort_month", "cohort_index"])["customer_id"].nunique().reset_index()
            retention = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="customer_id").divide(cohort_sizes, axis=0)

            revenue_data = of.groupby(["cohort_month", "cohort_index"])["order_value"].sum().reset_index()
            revenue_pivot = revenue_data.pivot(index="cohort_month", columns="cohort_index", values="order_value")
            avg_cum_clv = revenue_pivot.cumsum(axis=1).divide(cohort_sizes, axis=0)
            return retention, avg_cum_clv

        retention, avg_cum_clv = build_cohorts()

        retention_display = retention.iloc[:15, :13].copy()
        retention_display.index = retention_display.index.astype(str)
        fig = px.imshow(retention_display, text_auto=".0%", color_continuous_scale="Blues",
                        labels=dict(x="Months Since Signup", y="Signup Cohort", color="Retention"),
                        aspect="auto")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cumulative Revenue per Customer by Cohort")
        plot_data = avg_cum_clv.iloc[:8].reset_index()
        plot_data["cohort_month"] = plot_data["cohort_month"].astype(str)
        plot_long = plot_data.melt(id_vars="cohort_month", var_name="months_since_signup", value_name="cumulative_revenue")
        fig2 = px.line(plot_long, x="months_since_signup", y="cumulative_revenue", color="cohort_month", markers=True,
                       labels={"cumulative_revenue": "Cumulative Revenue per Customer (₹)", "months_since_signup": "Months Since Signup"})
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE 4 — A/B TEST SIMULATOR
# ============================================================
elif page == "A/B Test Simulator":
    st.title("A/B Test Simulator")
    st.caption("Design an experiment, run power analysis, simulate results, and get a business recommendation — live.")

    c1, c2, c3 = st.columns(3)
    baseline_cr = c1.slider("Baseline conversion rate", 0.02, 0.30, 0.12, step=0.01, format="%.2f")
    mde_relative = c2.slider("Minimum detectable lift (relative)", 0.05, 0.50, 0.15, step=0.05, format="%.2f")
    discount_aov_impact = c3.slider("Expected AOV impact of promotion", -0.30, 0.10, -0.10, step=0.05, format="%.2f")

    c4, c5 = st.columns(2)
    alpha = c4.selectbox("Significance level (alpha)", [0.01, 0.05, 0.10], index=1)
    power_target = c5.selectbox("Desired power", [0.70, 0.80, 0.90], index=1)

    run_button = st.button("🎲 Run Simulation", type="primary")

    if run_button:
        target_cr = baseline_cr * (1 + mde_relative)
        effect_size = 2 * (np.arcsin(np.sqrt(target_cr)) - np.arcsin(np.sqrt(baseline_cr)))
        analysis = NormalIndPower()
        n_per_group = int(np.ceil(analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power_target, ratio=1.0)))

        st.subheader("Step 1 — Power Analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Required sample size / group", f"{n_per_group:,}")
        c2.metric("Target conversion rate", f"{target_cr:.2%}")
        c3.metric("Total sample size", f"{n_per_group*2:,}")

        sample_sizes = np.arange(max(200, n_per_group // 10), n_per_group * 2, max(1, n_per_group // 40))
        powers = [analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0) for n in sample_sizes]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_sizes, y=powers, mode="lines", name="Power"))
        fig.add_hline(y=power_target, line_dash="dash", line_color="red", annotation_text=f"Target power = {power_target}")
        fig.add_vline(x=n_per_group, line_dash="dash", line_color="green", annotation_text=f"n = {n_per_group:,}")
        fig.update_layout(xaxis_title="Sample size per group", yaxis_title="Statistical Power")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Step 2 — Simulated Experiment")
        np.random.seed(42)
        control_conv = np.random.binomial(1, baseline_cr, n_per_group)
        treatment_conv = np.random.binomial(1, target_cr, n_per_group)
        control_aov = np.random.normal(1800, 450, max(control_conv.sum(), 1))
        treatment_aov = np.random.normal(1800 * (1 + discount_aov_impact), 430, max(treatment_conv.sum(), 1))

        control_cr_obs = control_conv.mean()
        treatment_cr_obs = treatment_conv.mean()

        count = np.array([treatment_conv.sum(), control_conv.sum()])
        nobs = np.array([n_per_group, n_per_group])
        z_stat, p_value_cr = proportions_ztest(count, nobs, alternative="larger")
        t_stat, p_value_aov, _ = ttest_ind(treatment_aov, control_aov, alternative="two-sided")

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(data=[go.Bar(x=["Control", "Treatment"], y=[control_cr_obs, treatment_cr_obs],
                                           marker_color=["#4C72B0", "#DD8452"],
                                           text=[f"{control_cr_obs:.2%}", f"{treatment_cr_obs:.2%}"], textposition="outside")])
            fig.update_layout(title=f"Conversion Rate (p={p_value_cr:.4f})", yaxis_title="Conversion Rate")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(data=[go.Bar(x=["Control", "Treatment"], y=[control_aov.mean(), treatment_aov.mean()],
                                           marker_color=["#4C72B0", "#DD8452"],
                                           text=[f"₹{control_aov.mean():,.0f}", f"₹{treatment_aov.mean():,.0f}"], textposition="outside")])
            fig.update_layout(title=f"Average Order Value (p={p_value_aov:.4f})", yaxis_title="AOV (₹)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Step 3 — Business Recommendation")
        revenue_control = control_cr_obs * control_aov.mean()
        revenue_treatment = treatment_cr_obs * treatment_aov.mean()
        lift = revenue_treatment - revenue_control

        col1, col2, col3 = st.columns(3)
        col1.metric("Revenue/visitor — Control", f"₹{revenue_control:.2f}")
        col2.metric("Revenue/visitor — Treatment", f"₹{revenue_treatment:.2f}")
        col3.metric("Net lift", f"₹{lift:.2f}", delta=f"{lift:.2f}")

        significant = p_value_cr < alpha
        if significant and lift > 0:
            st.success(
                "**Recommendation: Roll out.** The conversion lift is statistically significant "
                "and revenue-per-visitor improves overall. Monitor for novelty effects over the "
                "first 4-6 weeks and consider testing a smaller discount to protect margin further."
            )
        elif significant and lift <= 0:
            st.warning(
                "**Recommendation: Do not roll out as-is.** The conversion lift is statistically "
                "significant, but the drop in average order value outweighs it — net revenue per "
                "visitor is flat or negative. Consider a smaller discount or a non-price lever."
            )
        else:
            st.error(
                "**Recommendation: Inconclusive.** The observed difference is not statistically "
                "significant at this sample size / alpha level. Either collect more data or "
                "reconsider the minimum detectable effect you're testing for."
            )
    else:
        st.info("Adjust the assumptions above and click **Run Simulation** to see results.")
