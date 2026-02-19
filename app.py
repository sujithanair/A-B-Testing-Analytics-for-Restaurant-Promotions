# app.py
import streamlit as st
import pandas as pd

from campaign_analysis import (
    load_and_clean_data,
    basic_sales_stats,
    kpi_average_sales_by_promotion,
    check_normality_shapiro,
    check_levene_variance,
    run_anova,
    run_tukey_if_significant,
    make_sales_boxplot,
    make_avg_sales_barplot,
    save_fig,
    generate_executive_report_fallback,
)

st.set_page_config(page_title="Fast Food Campaign Dashboard", layout="wide")
st.title("Marketing Campaign Dashboard")

# ---- Config ----
DEFAULT_CSV = "dataset/WA_Marketing-Campaign.csv"
OUTPUT_DIR = "output"

# ---- Load ----
@st.cache_data
def cached_load(path: str) -> pd.DataFrame:
    return load_and_clean_data(path)

csv_path = st.sidebar.text_input("CSV path", value=DEFAULT_CSV)
df = cached_load(csv_path)

st.sidebar.header("Filters")
promos = sorted(df["Promotion"].unique())
selected_promos = st.sidebar.multiselect("Promotion", promos, default=promos)

df_f = df[df["Promotion"].isin(selected_promos)].copy()

# ---- KPIs / Stats ----
kpi = kpi_average_sales_by_promotion(df_f)
anova = run_anova(df_f)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df_f):,}")
c2.metric("Promotions", f"{df_f['Promotion'].nunique():,}")
c3.metric("ANOVA p-value", "N/A" if anova["p_value"] is None else f"{anova['p_value']:.6f}")

# ---- Tabs ----
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Assumptions", "Post-hoc", "Executive Report"])

with tab1:
    st.subheader("Basic Stats: SalesInThousands")
    st.dataframe(basic_sales_stats(df_f).to_frame("value"), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Sales Distribution by Promotion")
        fig_box = make_sales_boxplot(df_f)
        st.pyplot(fig_box, clear_figure=True)

    with right:
        st.subheader("Average Sales per Promotion")
        fig_bar = make_avg_sales_barplot(kpi)
        st.pyplot(fig_bar, clear_figure=True)

    st.subheader("KPI Table")
    st.dataframe(kpi.sort_values("AverageSales", ascending=False), use_container_width=True)

    if st.checkbox("Save charts to output/"):
        save_fig(fig_box, f"{OUTPUT_DIR}/sales_boxplot.png")
        save_fig(fig_bar, f"{OUTPUT_DIR}/avg_sales_bar.png")
        st.success("Saved charts to output/")

with tab2:
    st.subheader("Normality (Shapiro-Wilk) by Promotion")
    shapiro_df = check_normality_shapiro(df_f)
    st.dataframe(shapiro_df, use_container_width=True)

    st.subheader("Homogeneity of Variance (Levene)")
    levene = check_levene_variance(df_f)
    st.write(levene)

with tab3:
    st.subheader("ANOVA")
    if anova["p_value"] is None:
        st.warning("Not enough promotions selected to run ANOVA.")
    else:
        st.write({"F": anova["F"], "p_value": anova["p_value"]})

    st.subheader("Tukey HSD (only if ANOVA is significant)")
    if anova["p_value"] is not None and anova["p_value"] < 0.05:
        tukey_df = run_tukey_if_significant(df_f)
        if tukey_df is None:
            st.info("Not enough groups to run Tukey.")
        else:
            st.dataframe(tukey_df, use_container_width=True)
    else:
        st.info("ANOVA not significant (or not available), so Tukey is not shown.")

with tab4:
    st.markdown("### Executive Summary")
    
    report = generate_executive_report_fallback(kpi, anova)
    st.write(report)

    st.download_button(
        "Download report as .txt",
        data=report.encode("utf-8"),
        file_name="executive_report.txt",
        mime="text/plain",
    )