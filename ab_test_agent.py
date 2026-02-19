import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import openai


# 1. Load and clean data

df = pd.read_csv("dataset/WA_Marketing-Campaign.csv")

# Drop rows with missing values
df = df.dropna(subset=['SalesInThousands', 'Promotion'])

# Ensure sales are numeric
df['SalesInThousands'] = pd.to_numeric(df['SalesInThousands'], errors='coerce')
df = df.dropna(subset=['SalesInThousands'])


# 2. Basic statistics

print("Basic stats for SalesInThousands:")
print(df['SalesInThousands'].describe())


# 3. KPI: Average Sales per Promotion

kpi = df.groupby('Promotion')['SalesInThousands'].agg(['mean', 'count']).reset_index()
kpi.columns = ['Promotion', 'AverageSales', 'SampleSize']

print("\nAverage Sales per Promotion:")
print(kpi)


# 4. Check ANOVA assumptions

# Normality per group (Shapiro-Wilk)
print("\nNormality check (Shapiro-Wilk test) per promotion:")
for promo, sales in df.groupby('Promotion')['SalesInThousands']:
    stat, p = stats.shapiro(sales)
    print(f"Promotion {promo}: p-value = {p:.4f} {'(OK)' if p>0.05 else '(Check!)'}")

# Homogeneity of variance (Levene's test)
groups = [group['SalesInThousands'].values for name, group in df.groupby('Promotion')]
stat, p_levene = stats.levene(*groups)
print(f"\nLevene's test for equal variance: p-value = {p_levene:.4f} {'(OK)' if p_levene>0.05 else '(Check!)'}")


# 5. ANOVA test

f_stat, p_value = stats.f_oneway(*groups)
print(f"\nANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.10f}")

# Post-hoc test if significant
if p_value < 0.05:
    print("\nPost-hoc Tukey HSD test:")
    model = ols('SalesInThousands ~ C(Promotion)', data=df).fit()
    tukey = pairwise_tukeyhsd(endog=df['SalesInThousands'], groups=df['Promotion'], alpha=0.05)
    print(tukey)


# 6. Visualizations

if not os.path.exists("output"):
    os.makedirs("output")

# Boxplot of sales distribution
plt.figure(figsize=(8,6))
sns.boxplot(x='Promotion', y='SalesInThousands', data=df)
plt.title('Sales Distribution by Promotion')
plt.savefig("output/sales_boxplot.png")
plt.show()

# Bar chart of average sales
plt.figure(figsize=(8,6))
sns.barplot(x='Promotion', y='AverageSales', data=kpi, ci=None)
plt.title('Average Sales per Promotion')
plt.savefig("output/avg_sales_bar.png")
plt.show()


os.getenv("OPENAI_API_KEY")

# Load the key

if not openai.api_key:
    raise ValueError("Could not find 'key' in kaggle.json")

top_kpi = kpi.sort_values('AverageSales', ascending=False).head(5)
summary_text = "A/B test results for sales by promotion (top 5 shown):\n\n"
for idx, row in top_kpi.iterrows():
    summary_text += f"Promotion {row['Promotion']}: Average Sales = {row['AverageSales']:.1f} (n={row['SampleSize']})\n"

summary_text += f"\nANOVA F-statistic = {f_stat:.3f}, p-value = {p_value:.10f}\n"
summary_text += "Please provide a short executive summary (3-4 sentences) highlighting which promotion performs best, whether the differences are statistically significant, and recommended actions."

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set. Please export your API key first.")

# Prepare concise prompt (top 5 promotions only)
top_kpi = kpi.sort_values('AverageSales', ascending=False).head(5)

summary_text = "A/B test results for sales by promotion (top 5 shown):\n\n"

for _, row in top_kpi.iterrows():
    summary_text += (
        f"Promotion {row['Promotion']}: "
        f"Average Sales = {row['AverageSales']:.1f} "
        f"(n={row['SampleSize']})\n"
    )

summary_text += (
    f"\nANOVA F-statistic = {f_stat:.3f}, "
    f"p-value = {p_value:.10f}\n\n"
    "Please provide a short executive summary (3-4 sentences) "
    "highlighting which promotion performs best, "
    "whether differences are statistically significant, "
    "and recommended actions."
)

# Default fallback summary
def generate_fallback_summary():
    top_promo = top_kpi.iloc[0]
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

    return (
        "EXECUTIVE SUMMARY (Fallback Mode):\n\n"
        f"Promotion {top_promo['Promotion']} shows the highest average sales "
        f"({top_promo['AverageSales']:.1f}). "
        f"The ANOVA test indicates that differences between promotions are {significance} "
        f"(p = {p_value:.5f}). "
        "It is recommended to prioritize the top-performing promotion and consider "
        "additional controlled testing before scaling broadly."
    )

# Try OpenAI call
kpi_sorted = kpi.sort_values("AverageSales", ascending=False).reset_index(drop=True)

top_promo = kpi_sorted.iloc[0]
second_promo = kpi_sorted.iloc[1]

# Calculate lift vs second best
lift_vs_second = (
    (top_promo["AverageSales"] - second_promo["AverageSales"])
    / second_promo["AverageSales"]
) * 100

# Determine statistical strength
if p_value < 0.001:
    significance_level = "very strong statistical evidence"
elif p_value < 0.01:
    significance_level = "strong statistical evidence"
elif p_value < 0.05:
    significance_level = "moderate statistical evidence"
else:
    significance_level = "no statistically significant evidence"

# -----------------------------
# Prompt for OpenAI
# -----------------------------
summary_prompt = f"""
You are a senior marketing analytics consultant.

Results:
Top Promotion: {top_promo['Promotion']}
Top Avg Sales: {top_promo['AverageSales']:.2f}
Second Best Promotion: {second_promo['Promotion']}
Lift vs Second Best: {lift_vs_second:.2f}%
ANOVA p-value: {p_value:.6f}

Statistical interpretation: {significance_level}

Write a concise executive report including:
1. Clear winner
2. Business interpretation of lift
3. Statistical confidence explanation
4. Specific recommended next actions
Keep it executive-ready (5-6 sentences).
"""

# -----------------------------
# Fallback Report Generator
# -----------------------------
def fallback_report():
    confidence_text = (
        "The differences are statistically significant."
        if p_value < 0.05
        else "The differences are not statistically significant."
    )

    scale_recommendation = (
        "Recommend scaling this promotion immediately."
        if p_value < 0.05
        else "Recommend running additional controlled tests before scaling."
    )

    return f"""
EXECUTIVE REPORT (Fallback Mode)

Promotion {top_promo['Promotion']} delivers the highest average sales 
({top_promo['AverageSales']:.2f}), outperforming Promotion {second_promo['Promotion']} 
by {lift_vs_second:.2f}%.

ANOVA results (p = {p_value:.5f}) indicate {confidence_text}

Business Recommendation:
{scale_recommendation}
Additionally, consider validating results across more locations or time periods 
before full rollout.
"""

# -----------------------------
# Run AI Agent
# -----------------------------
try:
    if openai.api_key:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        ai_summary = response.choices[0].message.content
    else:
        raise Exception("API key missing")

except Exception as e:
    print("AI unavailable — switching to fallback.\n")
    ai_summary = fallback_report()

# -----------------------------
# Output
# -----------------------------
print("\n==============================")
print("EXECUTIVE ANALYSIS REPORT")
print("==============================\n")
print(ai_summary)

os.makedirs("output", exist_ok=True)

with open("output/executive_report.txt", "w") as f:
    f.write(ai_summary)