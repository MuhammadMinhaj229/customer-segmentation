# 🎯 Customer Segmentation & Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![MLflow](https://img.shields.io/badge/MLflow-2.8-purple)

> A self-directed project exploring two questions on a public telecom dataset: can customers be usefully segmented by behavior, and can churn be predicted early enough — and explained clearly enough — to matter?

---

## The Problem This Project Explores

Telecom and subscription businesses lose customers quietly — by the time someone cancels, the relationship is already over. This project uses the IBM Telco Customer Churn dataset (7,043 records) to explore three questions:

1. Can customers be grouped into meaningful behavioral segments instead of being treated as one group?
2. Can a model flag churn risk early, with enough accuracy to be useful?
3. Can the model explain *why* it flagged someone — not just produce a score?

This is a learning project built end-to-end on public data, not a system deployed at a real company — the goal was to do the full workflow properly and be honest about what it does and doesn't prove.

---

## Part 1 — RFM Customer Segmentation

Each customer is scored on three dimensions — Recency, Frequency, Monetary value — and K-Means clustering groups them into 5 segments:

| Segment | Profile | Share of Customers |
|---|---|---|
| Champions | High R, High F, High M | 18% |
| Loyal Customers | High F, medium M/R | 21% |
| At Risk | Formerly high-value, now inactive | 16% |
| Hibernating | Low activity, long inactive | 28% |
| Lost | No recent activity | 17% |

Segmentation quality: Silhouette score 0.61.

---

## Part 2 — Churn Prediction Model

**Architecture:** XGBoost Classifier, class imbalance corrected with SMOTE (raw churn rate 26.5%), hyperparameters tuned via GridSearchCV (50+ combinations), explainability via SHAP.

**Model performance (measured on held-out test data):**

| Metric | Score |
|---|---|
| AUC-ROC | 0.89 |
| Precision (Churn class) | 0.84 |
| Recall (Churn class) | 0.81 |
| F1 Score | 0.82 |
| Accuracy | 82.3% |

**Top churn drivers identified by SHAP:**
1. Contract type — month-to-month customers churn far more than 2-year contract customers
2. Tenure — customers in their first 12 months show the highest risk
3. Monthly charges, fiber-optic internet service, and lack of tech support also show strong effects

---

## Key Dataset Findings

- Churn rate in this dataset: 26.5%
- Month-to-month contracts account for the large majority of churned customers
- The first 12 months of tenure is the highest-risk retention window
- Fiber-optic customers churn at a notably higher rate than DSL customers in this dataset

(These are descriptive findings from this specific dataset — not claims about any live business.)

---

## Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Segmentation | Scikit-learn (K-Means) |
| Prediction | XGBoost |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Deployment | Streamlit |

---

## Project Structure
```
customer-segmentation/
├── data/                    # Telco churn dataset
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_rfm_analysis.ipynb
│   ├── 03_kmeans_clustering.ipynb
│   └── 04_churn_model.ipynb
├── app/
│   └── streamlit_app.py     # Live dashboard
├── requirements.txt
└── README.md
```

## How to Run Locally
```bash
git clone https://github.com/MuhammadMinhaj229/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Live Demo

👉 [Launch Dashboard](https://customer-segmentation-efb7gqwwg2sqdzpymxxbz5.streamlit.app/)

---

*Built by Mohammed Minhaj Mahmood*
[LinkedIn](https://linkedin.com/in/muhammadminhaj229) · [GitHub](https://github.com/MuhammadMinhaj229) · Hyderabad, India
