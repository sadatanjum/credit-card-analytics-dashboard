# 💳 Indian Credit Card Spending Analytics & ML Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-indigo.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> An end-to-end exploratory analytics and predictive machine learning platform analyzing **26,052 credit card transactions** across **986 Indian cities** (~₹4.07 Billion in total spend) between October 2013 and May 2015.

---

## 📌 Executive Summary & Key Analytical Findings

1. **Metro Concentration Effect (55.7% of Value)**:
   - Four metropolitan cities (**Bengaluru, Greater Mumbai, Ahmedabad, and Delhi**) drive **55.7% of total transaction value** (~₹2.27 Billion) and ~54% of all transaction volume.
   - The remaining 982 cities form a long tail with low transaction density.

2. **Demographic Asymmetry in Utility/Bill Payments**:
   - While overall transaction counts are balanced (52.5% Female, 47.5% Male), **Female cardholders exhibit significantly higher average spending on Bills/Utilities** (**₹202,810** average ticket size vs. **₹147,447** for males).
   - This points directly to high-ROI targeted utility cashback and rewards campaigns tailored for female primary cardholders.

3. **Card Tier Uniformity**:
   - Average spending amounts across card tiers are remarkably uniform (~**₹154K–₹157K** across Silver, Gold, Platinum, and Signature).
   - This demonstrates that card tiers in this market operate primarily as lifestyle branding and perk differentiators rather than strict credit limit gates.

---

## 🏗️ Project Architecture & Repository Structure

```
├── Credit card transactions - India - Simple.csv   # Primary dataset (26,052 rows, 7 columns)
├── analysis.ipynb                                  # Production-grade Jupyter notebook (EDA + ML with pre-rendered visuals)
├── app.py                                          # Interactive Streamlit dashboard with real-time inference
├── requirements.txt                                # Pinned dependencies for Streamlit Cloud & local setup
├── .streamlit/
│   └── config.toml                                 # Modern dashboard theme & server configuration
├── Project_group14.ipynb                           # Original baseline academic notebook (preserved)
└── README.md                                       # Comprehensive project documentation
```

---

## 🚀 Interactive Streamlit Dashboard Features

The web application (`app.py`) is structured into four interactive modules:

- **📊 Executive KPI Dashboard**: Real-time KPI metric cards (Total Spend, Transaction Count, Average Ticket Size, Active Cities) coupled with monthly spending velocity and category donut distributions.
- **🏙️ Geo & Demographic Deep-Dive**: Visualizations comparing Tier 1 Metros against emerging cities, gender spending asymmetry charts, and category $\times$ day-of-week heatmaps.
- **🤖 ML Insights & Live Predictor**:
  - Benchmark performance tables and normalized confusion matrices.
  - Interactive feature importance bar charts.
  - **Live Inference Widget**: Input transaction amount, card tier, gender, city tier, and day of week to get real-time expense category predictions and probability distributions.
- **💡 Strategic Insights & Portfolio Summary**: Structured business takeaways for banking product managers and loyalty program designers.

---

## 🤖 Machine Learning Pipeline & Methodology

### 1. Problem Formulation
- **Objective**: Multiclass classification to predict a transaction's **Expense Category** (`Exp Type`: *Bills, Food, Fuel, Entertainment, Grocery, Travel*) from financial magnitude and contextual metadata.
- **Features Used**: `Amount`, `Card Type`, `Gender`, `City_Tier` (Metros / Emerging / Other), `DayOfWeek`, `Month`, `IsWeekend`.

### 2. Preprocessing & Engineering
- `ColumnTransformer` with `StandardScaler` for continuous features (`Amount`, `Month`, `IsWeekend`).
- `OneHotEncoder(drop='first', handle_unknown='ignore')` for categorical attributes.
- Stratified 80/20 train/test split.

### 3. Model Benchmark (Test Set)

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **20.03%** | **31.16%** | **0.188** | **0.201** |
| **Random Forest Classifier** | 19.25% | 26.90% | 0.177 | 0.194 |
| **HistGradientBoosting** | 21.84% | 19.35% | 0.183 | 0.208 |

*Note: With 6 target categories, random baseline accuracy is ~16.7%. Logistic Regression with class-weight balancing achieves the highest balanced accuracy across minority categories (such as Travel at 2.8% prevalence).*

---

## 🛠️ Local Installation & Running the App

### Prerequisites
- Python 3.10, 3.11, or 3.12+

### Step-by-Step Setup

1. **Clone the project repository**:
   ```bash
   git clone https://github.com/sadatanjum/credit-card-analytics-dashboard.git
   cd credit-card-analytics-dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   The dashboard will automatically open in your default browser at `http://localhost:8501`.

4. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook analysis.ipynb
   ```

---

## ☁️ Deploying to Streamlit Community Cloud (Free)

1. **Create a GitHub Repository**:
   - Push this project folder (`Credit card transactions - India - Simple.csv`, `app.py`, `requirements.txt`, `.streamlit/config.toml`, `README.md`) to your GitHub account.

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
   - Click **"New app"**.
   - Select your repository, branch (`main`), and set the main file path to `app.py`.
   - Click **"Deploy!"**.

Your live portfolio application will be online with a shareable public URL in under 2 minutes!

---

## 📈 Engineering Journey & Improvements

| Area | Initial Academic Baseline | Refactored Portfolio Solution |
| :--- | :--- | :--- |
| **Data Integrity** | Overwrote 26k-row dataset with a 5-row toy dataframe | Full 26,052-row dataset with automated schema & null validation |
| **Geographic Handling** | Sparse 986-city label encoding (unviable) | Engineered 3-Tier hierarchy (Metros, Emerging, Other) |
| **Problem Formulation** | Regression attempting to predict random amounts ($R^2 < 0$) | Statistically grounded multiclass classification & behavioral profiling |
| **Evaluation Metrics** | Hardcoded/simulated precision & recall scores | Real Stratified K-Fold CV, Macro F1, and normalized confusion matrices |
| **Deliverable** | Static broken notebook | Interactive Streamlit web app + fully executed `analysis.ipynb` |

---

## 📜 License
This project is licensed under the **MIT License** — free to use for personal and commercial portfolio demonstrations.
