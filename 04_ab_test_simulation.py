"""
04_ab_test_simulation.py
---------------------------
End-to-end A/B test workflow for a hypothetical business question:

  "If we offer a 10% discount promotion at checkout, does it increase
   the conversion rate (visit -> purchase) enough to be worth the
   margin hit?"

Steps:
  1. Power analysis -> how many visitors do we need per group?
  2. Simulate a realistic experiment (control vs treatment) using a
     known ground-truth effect size, so we can sanity-check that our
     test correctly detects it.
  3. Run the statistical test (two-proportion z-test) + a secondary
     t-test on average order value.
  4. Translate results into a business recommendation ("now what").
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.weightstats import ttest_ind
import scipy.stats as stats

sns.set_style("whitegrid")
np.random.seed(42)
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# =========================================================
# 1. POWER ANALYSIS — how many samples do we need?
# =========================================================
baseline_cr = 0.12          # current checkout conversion rate (from historical data)
mde_relative = 0.15         # minimum detectable effect: we care about a 15% relative lift
target_cr = baseline_cr * (1 + mde_relative)   # 0.138

# Cohen's h effect size for two proportions
effect_size = 2 * (np.arcsin(np.sqrt(target_cr)) - np.arcsin(np.sqrt(baseline_cr)))

alpha = 0.05
power = 0.80

analysis = NormalIndPower()
n_per_group = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
n_per_group = int(np.ceil(n_per_group))

print("===== POWER ANALYSIS =====")
print(f"Baseline conversion rate: {baseline_cr:.1%}")
print(f"Target conversion rate (15% relative lift): {target_cr:.1%}")
print(f"Effect size (Cohen's h): {effect_size:.4f}")
print(f"Required sample size per group (alpha=0.05, power=0.80): {n_per_group:,}")
print(f"Total sample size needed: {n_per_group * 2:,}")

# Power curve across a range of sample sizes (for the report/visual)
sample_sizes = np.arange(500, 20000, 250)
powers = [analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0) for n in sample_sizes]

plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, powers, linewidth=2)
plt.axhline(0.8, color="red", linestyle="--", label="Target power = 0.80")
plt.axvline(n_per_group, color="green", linestyle="--", label=f"Required n = {n_per_group:,}")
plt.xlabel("Sample size per group")
plt.ylabel("Statistical Power")
plt.title("Power Curve: Detecting a 15% Relative Lift in Conversion Rate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/ab_test_power_curve.png", dpi=150)
plt.close()

# =========================================================
# 2. SIMULATE THE EXPERIMENT
# =========================================================
# We simulate with the SAME true effect used in the power analysis, to
# show the test correctly detects what we designed it to detect.
n_control = n_per_group
n_treatment = n_per_group

control_conversions = np.random.binomial(1, baseline_cr, n_control)
treatment_conversions = np.random.binomial(1, target_cr, n_treatment)

# Simulate order values (only for converters) — treatment group has slightly
# lower AOV because of the discount, a realistic trade-off to discuss
control_aov = np.random.normal(1800, 450, control_conversions.sum())
treatment_aov = np.random.normal(1650, 430, treatment_conversions.sum())  # discount reduces AOV

# =========================================================
# 3. RUN STATISTICAL TESTS
# =========================================================
print("\n===== EXPERIMENT RESULTS =====")
control_cr = control_conversions.mean()
treatment_cr = treatment_conversions.mean()
print(f"Control conversion rate:   {control_cr:.2%}  (n={n_control:,})")
print(f"Treatment conversion rate: {treatment_cr:.2%}  (n={n_treatment:,})")
print(f"Absolute lift: {(treatment_cr - control_cr):.2%}")
print(f"Relative lift: {(treatment_cr - control_cr) / control_cr:.1%}")

# Two-proportion z-test
count = np.array([treatment_conversions.sum(), control_conversions.sum()])
nobs = np.array([n_treatment, n_control])
z_stat, p_value_cr = proportions_ztest(count, nobs, alternative="larger")
ci_low, ci_high = proportion_confint(count[0], nobs[0], alpha=0.05, method="wilson")

print(f"\nTwo-proportion z-test (treatment > control):")
print(f"  z-statistic = {z_stat:.3f}, p-value = {p_value_cr:.5f}")
print(f"  95% CI for treatment conversion rate: [{ci_low:.2%}, {ci_high:.2%}]")
print(f"  Statistically significant at alpha=0.05: {'YES' if p_value_cr < alpha else 'NO'}")

# t-test on AOV (does the discount hurt average order value significantly?)
t_stat, p_value_aov, dof = ttest_ind(treatment_aov, control_aov, alternative="two-sided")
print(f"\nIndependent t-test on Average Order Value (treatment vs control):")
print(f"  Control AOV mean: Rs.{control_aov.mean():,.0f} | Treatment AOV mean: Rs.{treatment_aov.mean():,.0f}")
print(f"  t-statistic = {t_stat:.3f}, p-value = {p_value_aov:.5f}")
print(f"  Statistically significant AOV difference: {'YES' if p_value_aov < alpha else 'NO'}")

# =========================================================
# 4. BUSINESS IMPACT ("now what")
# =========================================================
incremental_conversions_per_1000_visitors = (treatment_cr - control_cr) * 1000
revenue_control = control_cr * control_aov.mean()      # expected revenue per visitor, control
revenue_treatment = treatment_cr * treatment_aov.mean()  # expected revenue per visitor, treatment
revenue_lift_per_visitor = revenue_treatment - revenue_control

print("\n===== BUSINESS IMPACT =====")
print(f"Expected revenue per visitor — Control:   Rs.{revenue_control:.2f}")
print(f"Expected revenue per visitor — Treatment: Rs.{revenue_treatment:.2f}")
print(f"Net revenue-per-visitor lift: Rs.{revenue_lift_per_visitor:.2f} "
      f"({'positive' if revenue_lift_per_visitor > 0 else 'negative'} despite lower AOV)")

# Visualization: conversion rate comparison with CI
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

groups = ["Control", "Treatment"]
crs = [control_cr, treatment_cr]
ci_control = proportion_confint(control_conversions.sum(), n_control, alpha=0.05, method="wilson")
ci_treatment = (ci_low, ci_high)
errors = [[crs[0] - ci_control[0], crs[1] - ci_treatment[0]],
          [ci_control[1] - crs[0], ci_treatment[1] - crs[1]]]

axes[0].bar(groups, crs, yerr=errors, capsize=8, color=["#4C72B0", "#DD8452"])
axes[0].set_ylabel("Conversion Rate")
axes[0].set_title(f"Conversion Rate by Group\n(p={p_value_cr:.4f})")
for i, v in enumerate(crs):
    axes[0].text(i, v + 0.003, f"{v:.2%}", ha="center", fontweight="bold")

axes[1].bar(groups, [control_aov.mean(), treatment_aov.mean()],
            yerr=[control_aov.std()/np.sqrt(len(control_aov)), treatment_aov.std()/np.sqrt(len(treatment_aov))],
            capsize=8, color=["#4C72B0", "#DD8452"])
axes[1].set_ylabel("Average Order Value (Rs.)")
axes[1].set_title(f"AOV by Group\n(p={p_value_aov:.4f})")
for i, v in enumerate([control_aov.mean(), treatment_aov.mean()]):
    axes[1].text(i, v + 20, f"Rs.{v:,.0f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/ab_test_results.png", dpi=150)
plt.close()

# Save a clean summary table
summary = pd.DataFrame([
    {"metric": "Baseline conversion rate", "value": f"{baseline_cr:.2%}"},
    {"metric": "Target conversion rate (design)", "value": f"{target_cr:.2%}"},
    {"metric": "Required sample size per group", "value": f"{n_per_group:,}"},
    {"metric": "Observed control CR", "value": f"{control_cr:.2%}"},
    {"metric": "Observed treatment CR", "value": f"{treatment_cr:.2%}"},
    {"metric": "Conversion z-test p-value", "value": f"{p_value_cr:.5f}"},
    {"metric": "AOV t-test p-value", "value": f"{p_value_aov:.5f}"},
    {"metric": "Revenue-per-visitor lift", "value": f"Rs.{revenue_lift_per_visitor:.2f}"},
])
summary.to_csv(f"{OUT}/ab_test_summary.csv", index=False)

print("\nSaved: ab_test_power_curve.png, ab_test_results.png, ab_test_summary.csv")
print("\n===== RECOMMENDATION (\"now what\") =====")
if p_value_cr < alpha and revenue_lift_per_visitor > 0:
    print("Roll out the discount promotion: the conversion lift is statistically")
    print("significant AND revenue-per-visitor improves despite the lower AOV.")
    print("Recommend monitoring for novelty effects over the first 4-6 weeks post-launch,")
    print("and re-testing with a smaller discount tier to find the margin-optimal point.")
else:
    print("Do not roll out as-is: either the lift isn't statistically significant,")
    print("or it comes at a net revenue loss per visitor. Consider testing a smaller")
    print("discount or a different lever (free shipping, loyalty points) instead.")
