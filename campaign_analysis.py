# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json 
# import os
# from scipy import stats
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# import openai


# # 1. Load and clean data

# df = pd.read_csv("dataset/WA_Marketing-Campaign.csv")

# # Drop rows with missing values
# df = df.dropna(subset=['SalesInThousands', 'Promotion'])

# # Ensure sales are numeric
# df['SalesInThousands'] = pd.to_numeric(df['SalesInThousands'], errors='coerce')
# df = df.dropna(subset=['SalesInThousands'])


# # 2. Basic statistics

# print("Basic stats for SalesInThousands:")
# print(df['SalesInThousands'].describe())


# # 3. KPI: Average Sales per Promotion

# kpi = df.groupby('Promotion')['SalesInThousands'].agg(['mean', 'count']).reset_index()
# kpi.columns = ['Promotion', 'AverageSales', 'SampleSize']

# print("\nAverage Sales per Promotion:")
# print(kpi)


# # 4. Check ANOVA assumptions

# # Normality per group (Shapiro-Wilk)
# print("\nNormality check (Shapiro-Wilk test) per promotion:")
# for promo, sales in df.groupby('Promotion')['SalesInThousands']:
#     stat, p = stats.shapiro(sales)
#     print(f"Promotion {promo}: p-value = {p:.4f} {'(OK)' if p>0.05 else '(Check!)'}")

# # Homogeneity of variance (Levene's test)
# groups = [group['SalesInThousands'].values for name, group in df.groupby('Promotion')]
# stat, p_levene = stats.levene(*groups)
# print(f"\nLevene's test for equal variance: p-value = {p_levene:.4f} {'(OK)' if p_levene>0.05 else '(Check!)'}")


# # 5. ANOVA test

# f_stat, p_value = stats.f_oneway(*groups)
# print(f"\nANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.10f}")

# # Post-hoc test if significant
# if p_value < 0.05:
#     print("\nPost-hoc Tukey HSD test:")
#     model = ols('SalesInThousands ~ C(Promotion)', data=df).fit()
#     tukey = pairwise_tukeyhsd(endog=df['SalesInThousands'], groups=df['Promotion'], alpha=0.05)
#     print(tukey)


# # 6. Visualizations

# if not os.path.exists("output"):
#     os.makedirs("output")

# # Boxplot of sales distribution
# plt.figure(figsize=(8,6))
# sns.boxplot(x='Promotion', y='SalesInThousands', data=df)
# plt.title('Sales Distribution by Promotion')
# plt.savefig("output/sales_boxplot.png")
# plt.show()

# # Bar chart of average sales
# plt.figure(figsize=(8,6))
# sns.barplot(x='Promotion', y='AverageSales', data=kpi, ci=None)
# plt.title('Average Sales per Promotion')
# plt.savefig("output/avg_sales_bar.png")
# plt.show()


# os.getenv("OPENAI_API_KEY")

# # Load the key

# if not openai.api_key:
#     raise ValueError("Could not find 'key' in kaggle.json")

# top_kpi = kpi.sort_values('AverageSales', ascending=False).head(5)
# summary_text = "A/B test results for sales by promotion (top 5 shown):\n\n"
# for idx, row in top_kpi.iterrows():
#     summary_text += f"Promotion {row['Promotion']}: Average Sales = {row['AverageSales']:.1f} (n={row['SampleSize']})\n"

# summary_text += f"\nANOVA F-statistic = {f_stat:.3f}, p-value = {p_value:.10f}\n"
# summary_text += "Please provide a short executive summary (3-4 sentences) highlighting which promotion performs best, whether the differences are statistically significant, and recommended actions."

# openai.api_key = os.getenv("OPENAI_API_KEY")

# if not openai.api_key:
#     raise ValueError("OPENAI_API_KEY not set. Please export your API key first.")

# # Prepare concise prompt (top 5 promotions only)
# top_kpi = kpi.sort_values('AverageSales', ascending=False).head(5)

# summary_text = "A/B test results for sales by promotion (top 5 shown):\n\n"

# for _, row in top_kpi.iterrows():
#     summary_text += (
#         f"Promotion {row['Promotion']}: "
#         f"Average Sales = {row['AverageSales']:.1f} "
#         f"(n={row['SampleSize']})\n"
#     )

# summary_text += (
#     f"\nANOVA F-statistic = {f_stat:.3f}, "
#     f"p-value = {p_value:.10f}\n\n"
#     "Please provide a short executive summary (3-4 sentences) "
#     "highlighting which promotion performs best, "
#     "whether differences are statistically significant, "
#     "and recommended actions."
# )

# # Default fallback summary
# def generate_fallback_summary():
#     top_promo = top_kpi.iloc[0]
#     significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

#     return (
#         "EXECUTIVE SUMMARY (Fallback Mode):\n\n"
#         f"Promotion {top_promo['Promotion']} shows the highest average sales "
#         f"({top_promo['AverageSales']:.1f}). "
#         f"The ANOVA test indicates that differences between promotions are {significance} "
#         f"(p = {p_value:.5f}). "
#         "It is recommended to prioritize the top-performing promotion and consider "
#         "additional controlled testing before scaling broadly."
#     )

# # Try OpenAI call
# kpi_sorted = kpi.sort_values("AverageSales", ascending=False).reset_index(drop=True)

# top_promo = kpi_sorted.iloc[0]
# second_promo = kpi_sorted.iloc[1]

# # Calculate lift vs second best
# lift_vs_second = (
#     (top_promo["AverageSales"] - second_promo["AverageSales"])
#     / second_promo["AverageSales"]
# ) * 100

# # Determine statistical strength
# if p_value < 0.001:
#     significance_level = "very strong statistical evidence"
# elif p_value < 0.01:
#     significance_level = "strong statistical evidence"
# elif p_value < 0.05:
#     significance_level = "moderate statistical evidence"
# else:
#     significance_level = "no statistically significant evidence"

# # -----------------------------
# # Prompt for OpenAI
# # -----------------------------
# summary_prompt = f"""
# You are a senior marketing analytics consultant.

# Results:
# Top Promotion: {top_promo['Promotion']}
# Top Avg Sales: {top_promo['AverageSales']:.2f}
# Second Best Promotion: {second_promo['Promotion']}
# Lift vs Second Best: {lift_vs_second:.2f}%
# ANOVA p-value: {p_value:.6f}

# Statistical interpretation: {significance_level}

# Write a concise executive report including:
# 1. Clear winner
# 2. Business interpretation of lift
# 3. Statistical confidence explanation
# 4. Specific recommended next actions
# Keep it executive-ready (5-6 sentences).
# """

# # -----------------------------
# # Fallback Report Generator
# # -----------------------------
# def fallback_report():
#     confidence_text = (
#         "The differences are statistically significant."
#         if p_value < 0.05
#         else "The differences are not statistically significant."
#     )

#     scale_recommendation = (
#         "Recommend scaling this promotion immediately."
#         if p_value < 0.05
#         else "Recommend running additional controlled tests before scaling."
#     )

#     return f"""
# EXECUTIVE REPORT (Fallback Mode)

# Promotion {top_promo['Promotion']} delivers the highest average sales 
# ({top_promo['AverageSales']:.2f}), outperforming Promotion {second_promo['Promotion']} 
# by {lift_vs_second:.2f}%.

# ANOVA results (p = {p_value:.5f}) indicate {confidence_text}

# Business Recommendation:
# {scale_recommendation}
# Additionally, consider validating results across more locations or time periods 
# before full rollout.
# """

# # -----------------------------
# # Run AI Agent
# # -----------------------------
# try:
#     if openai.api_key:
#         response = openai.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": summary_prompt}]
#         )
#         ai_summary = response.choices[0].message.content
#     else:
#         raise Exception("API key missing")

# except Exception as e:
#     print("AI unavailable — switching to fallback.\n")
#     ai_summary = fallback_report()

# # -----------------------------
# # Output
# # -----------------------------
# print("\n==============================")
# print("EXECUTIVE ANALYSIS REPORT")
# print("==============================\n")
# print(ai_summary)

# os.makedirs("output", exist_ok=True)

# with open("output/executive_report.txt", "w") as f:
#     f.write(ai_summary)

# campaign_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# -----------------------------
# 1) Load and clean
# -----------------------------
def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop rows with missing values
    df = df.dropna(subset=["SalesInThousands", "Promotion"]).copy()

    # Ensure sales are numeric
    df["SalesInThousands"] = pd.to_numeric(df["SalesInThousands"], errors="coerce")
    df = df.dropna(subset=["SalesInThousands"]).copy()

    return df


# -----------------------------
# 2) Basic stats
# -----------------------------
def basic_sales_stats(df: pd.DataFrame) -> pd.Series:
    return df["SalesInThousands"].describe()


# -----------------------------
# 3) KPI: Avg sales per promotion
# -----------------------------
def kpi_average_sales_by_promotion(df: pd.DataFrame) -> pd.DataFrame:
    kpi = df.groupby("Promotion")["SalesInThousands"].agg(["mean", "count"]).reset_index()
    kpi.columns = ["Promotion", "AverageSales", "SampleSize"]
    return kpi


# -----------------------------
# 4) Assumption checks
# -----------------------------
def check_normality_shapiro(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for promo, grp in df.groupby("Promotion"):
        stat, p = stats.shapiro(grp["SalesInThousands"])
        rows.append(
            {"Promotion": promo, "ShapiroW": float(stat), "p_value": float(p), "Normal_p>0.05": p > 0.05}
        )
    return pd.DataFrame(rows).sort_values("Promotion")


def check_levene_variance(df: pd.DataFrame) -> dict:
    groups = [g["SalesInThousands"].values for _, g in df.groupby("Promotion")]
    stat, p = stats.levene(*groups)
    return {"LeveneW": float(stat), "p_value": float(p), "EqualVar_p>0.05": p > 0.05}


# -----------------------------
# 5) ANOVA + posthoc
# -----------------------------
def run_anova(df: pd.DataFrame) -> dict:
    groups = [g["SalesInThousands"].values for _, g in df.groupby("Promotion")]

    # Needs at least 2 groups
    if len(groups) < 2:
        return {"F": None, "p_value": None}

    f_stat, p_value = stats.f_oneway(*groups)
    return {"F": float(f_stat), "p_value": float(p_value)}


def run_tukey_if_significant(df: pd.DataFrame, alpha: float = 0.05):
    # Only run if enough groups
    if df["Promotion"].nunique() < 2:
        return None

    # Tukey itself doesn't require ANOVA result, but we gate it upstream in Streamlit.
    tukey = pairwise_tukeyhsd(endog=df["SalesInThousands"], groups=df["Promotion"], alpha=alpha)

    # Convert summary to DataFrame
    summary = tukey.summary()
    tukey_df = pd.DataFrame(summary.data[1:], columns=summary.data[0])
    return tukey_df


# -----------------------------
# 6) Visualizations (return figures)
# -----------------------------
def make_sales_boxplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="Promotion", y="SalesInThousands", data=df, ax=ax)
    ax.set_title("Sales Distribution by Promotion")
    fig.tight_layout()
    return fig


def make_avg_sales_barplot(kpi: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Promotion", y="AverageSales", data=kpi, ci=None, ax=ax)
    ax.set_title("Average Sales per Promotion")
    fig.tight_layout()
    return fig


def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


# -----------------------------
# 7) Executive report (fallback only)
# -----------------------------
def generate_executive_report_fallback(kpi: pd.DataFrame, anova: dict) -> str:
    if kpi.empty:
        return "EXECUTIVE REPORT\n\nNo data available after cleaning/filtering."

    kpi_sorted = kpi.sort_values("AverageSales", ascending=False).reset_index(drop=True)
    top = kpi_sorted.iloc[0]

    if len(kpi_sorted) > 1:
        second = kpi_sorted.iloc[1]
        lift_vs_second = ((top["AverageSales"] - second["AverageSales"]) / second["AverageSales"]) * 100
        lift_text = f"outperforming Promotion {second['Promotion']} by {lift_vs_second:.2f}%."
    else:
        lift_text = "with no comparable second-best promotion in the current selection."

    p_value = anova.get("p_value", None)
    if p_value is None:
        confidence_text = "ANOVA could not be computed with the current selection."
        scale_recommendation = "Recommend ensuring at least two promotions are included."
    else:
        confidence_text = (
            "The differences are statistically significant." if p_value < 0.05
            else "The differences are not statistically significant."
        )
        scale_recommendation = (
            "Recommend scaling this promotion immediately." if p_value < 0.05
            else "Recommend running additional controlled tests before scaling."
        )

    return f"""
EXECUTIVE REPORT (Fallback Mode)

Promotion {top['Promotion']} delivers the highest average sales ({top['AverageSales']:.2f}),
{lift_text}

ANOVA results: F = {anova.get('F')}, p = {p_value}
Interpretation: {confidence_text}

Business Recommendation:
{scale_recommendation}
Additionally, validate results across more locations/time periods before full rollout.
""".strip()