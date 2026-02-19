📊 Marketing Promotion Effectiveness Dashboard

An end-to-end A/B testing analytics project evaluating the impact of marketing promotions on sales performance for a fast food chain.
This project combines statistical rigor, business interpretation, and automated reporting to support data-driven promotion decisions.

🚀 Project Overview

Marketing teams frequently run promotions but struggle to determine:

	Which promotion truly performs best? 
	Are differences statistically significant?
	Should the winning campaign be scaled?

This project answers those questions using structured statistical analysis and an interactive dashboard.

🧠 Business Problem

A fast food chain conducted multiple promotional campaigns across locations.

The objective:

	Identify the highest-performing promotion
	Determine if performance differences are statistically significant
	Quantify lift between top campaigns
	Provide executive-ready recommendations

📈 Methodology

1️⃣ Data Cleaning & Validation

		Removed missing values
		Ensured numeric integrity of sales data

2️⃣ Statistical Assumption Testing

	Shapiro-Wilk Test for normality
	Levene’s Test for homogeneity of variance

3️⃣ Hypothesis Testing

	One-way ANOVA to detect differences in mean sales
	Tukey HSD post-hoc analysis when significant

4️⃣ Lift Calculation

	Percentage lift of top promotion vs second-best

5️⃣ Executive Reporting

	Automated summary generation
	Deterministic fallback logic in case of API failure

📊 Dashboard Features

	Upload campaign dataset
	View descriptive statistics
	Interactive boxplots and bar charts
	ANOVA test metrics (F-statistic & p-value)
	Lift vs second-best calculation
	Executive summary with business recommendations

🛠 Tech Stack

	Python
	pandas
	numpy
	scipy
	statsmodels
	matplotlib / seaborn
	Streamlit (for dashboard deployment)
	OpenAI API (optional AI-generated executive summary with fallback logic)

🏗 Architecture Highlights

1. Secure environment variable management for API keys

2. API failure handling with fallback summary generation

3. Automated report export to text file

4. Modular statistical pipeline

5. Reproducible and deployment-ready structure

📂 Project Structure

├── app.py                     # Streamlit dashboard

├── analyze_campaign.py        # Statistical analysis script

├── dataset/

│   └── WA_Marketing-Campaign.csv

├── output/

│   ├── sales_boxplot.png

│   ├── avg_sales_bar.png

│   └── executive_report.txt

├── requirements.txt

└── README.md

▶️ How to Run Locally
1. Clone the repository
git clone https://github.com/yourusername/Repository_Name.git

cd Repository_Name

2. Install dependencies
 
pip install -r requirements.txt

3. Run the dashboard
   
streamlit run app.py

📌 Key Insights Demonstrated

1. Translating statistical output into business recommendations

2. Understanding and validating ANOVA assumptions

3. Designing analytics workflows with production reliability in mind

4. Bridging data science with executive communication
