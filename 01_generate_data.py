"""
01_generate_data.py
--------------------
Generates a realistic synthetic e-commerce dataset:
 - customers.csv   : customer master data
 - orders.csv       : order-level transactions
 - order_items.csv  : line items per order (product, qty, price)

This simulates ~2 years of transaction history for ~5,000 customers.
Replace this with your real dataset if you have one (e.g. from Kaggle's
"Online Retail" or "E-Commerce Data" sets) — the rest of the pipeline
only expects the same column names, so it will still work.
"""

import os
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random

os.makedirs("data", exist_ok=True)

np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

N_CUSTOMERS = 5000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 6, 30)
CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Beauty",
              "Sports", "Books", "Grocery", "Toys"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
          "Pune", "Kolkata", "Ahmedabad", "Patna", "Jaipur"]

# ---------- 1. Customers ----------
customers = []
for cid in range(1, N_CUSTOMERS + 1):
    signup_date = fake.date_between(start_date=START_DATE, end_date=END_DATE - timedelta(days=30))
    # Assign a "customer type" that drives their future purchase behavior
    segment = np.random.choice(
        ["high_value", "regular", "low_engagement", "one_time"],
        p=[0.10, 0.45, 0.30, 0.15]
    )
    customers.append({
        "customer_id": cid,
        "signup_date": signup_date,
        "city": random.choice(CITIES),
        "age": np.random.randint(18, 65),
        "gender": random.choice(["M", "F"]),
        "segment": segment,  # hidden ground-truth driver, not used directly as a feature
        "acquisition_channel": np.random.choice(
            ["Organic", "Paid Ads", "Referral", "Email", "Social Media"],
            p=[0.30, 0.25, 0.15, 0.15, 0.15]
        )
    })
customers_df = pd.DataFrame(customers)

# ---------- 2. Orders & Order Items ----------
segment_params = {
    # (avg orders/month while active, churn probability per month, avg order value mean/std)
    "high_value":      dict(rate=2.2, churn_p=0.03, aov_mean=3200, aov_std=900),
    "regular":         dict(rate=1.0, churn_p=0.07, aov_mean=1600, aov_std=500),
    "low_engagement":  dict(rate=0.4, churn_p=0.15, aov_mean=900,  aov_std=300),
    "one_time":        dict(rate=0.15, churn_p=0.60, aov_mean=1100, aov_std=400),
}

orders = []
order_items = []
order_id_counter = 1
item_id_counter = 1

for _, cust in customers_df.iterrows():
    params = segment_params[cust["segment"]]
    signup = pd.to_datetime(cust["signup_date"])
    current_date = signup
    active = True
    while active and current_date < END_DATE:
        # decide if customer churns this month
        if np.random.random() < params["churn_p"]:
            active = False
            break
        # number of orders this month (Poisson around segment rate)
        n_orders_this_month = np.random.poisson(params["rate"])
        for _ in range(n_orders_this_month):
            order_date = current_date + timedelta(days=np.random.randint(0, 28))
            if order_date > END_DATE:
                continue
            order_value = max(200, np.random.normal(params["aov_mean"], params["aov_std"]))
            n_items = np.random.randint(1, 5)
            orders.append({
                "order_id": order_id_counter,
                "customer_id": cust["customer_id"],
                "order_date": order_date.date(),
                "order_value": round(order_value, 2),
                "discount_applied": np.random.choice([0, 1], p=[0.7, 0.3]),
                "payment_method": np.random.choice(["Card", "UPI", "COD", "Wallet"], p=[0.35, 0.35, 0.15, 0.15]),
            })
            remaining = order_value
            for i in range(n_items):
                cat = random.choice(CATEGORIES)
                item_val = round(remaining / (n_items - i) * np.random.uniform(0.7, 1.3), 2)
                remaining -= item_val
                order_items.append({
                    "order_item_id": item_id_counter,
                    "order_id": order_id_counter,
                    "category": cat,
                    "quantity": np.random.randint(1, 4),
                    "item_price": max(50, item_val),
                })
                item_id_counter += 1
            order_id_counter += 1
        current_date += timedelta(days=30)

orders_df = pd.DataFrame(orders)
order_items_df = pd.DataFrame(order_items)

# Drop the "segment" ground truth before saving customers (it's the latent variable
# our churn model will have to discover indirectly via behavior — keeping it would leak the answer)
customers_export = customers_df.drop(columns=["segment"])

customers_export.to_csv("data/customers.csv", index=False)
orders_df.to_csv("data/orders.csv", index=False)
order_items_df.to_csv("data/order_items.csv", index=False)

print(f"Customers: {len(customers_export)}")
print(f"Orders: {len(orders_df)}")
print(f"Order items: {len(order_items_df)}")
print(f"Date range: {orders_df['order_date'].min()} to {orders_df['order_date'].max()}")
