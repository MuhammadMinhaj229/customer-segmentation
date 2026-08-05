# Customer Segmentation & Churn Prediction

A machine learning project exploring customer behavioral segmentation and churn prediction using the IBM Telco Customer Churn dataset and an online retail dataset.

---

## Problem & Motivation

Subscription businesses often lose customers before they can intervene. This project explores two core objectives using public datasets:
1. Identifying distinct behavioral segments of customers using transactional data.
2. Building a model to predict which customers are at risk of churning, allowing for proactive retention efforts.

---

## Approach

1. **Data Processing:** Categorical features were encoded and class imbalances in the churn dataset were addressed using SMOTE.
2. **Segmentation (RFM):** Transactional data was processed to calculate Recency, Frequency, and Monetary (RFM) metrics. K-Means clustering was applied to group customers into 5 segments.
3. **Churn Prediction:** An XGBoost classifier was trained on the processed telecom dataset to predict customer churn probability.
4. **API & Dashboard:** The trained model is served via a FastAPI backend, and an interactive Streamlit dashboard allows for real-time customer risk profiling.

---

## Results

**RFM Segmentation**
- **Algorithm:** K-Means (5 clusters)
- **Silhouette Score:** 0.33 (Verified on `online_retail_II.csv`)

**Churn Prediction Model**
- **Algorithm:** XGBoost with SMOTE
- **Test Set Performance:**
  - **AUC-ROC:** 0.7977
  - **Precision:** 0.5185
  - **Recall:** 0.5615
  - **F1 Score:** 0.5392
  - **Accuracy:** 74.48%

---

## Tech Stack

- **Data Processing:** Python, Pandas, NumPy
- **Modeling:** Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **API Backend:** FastAPI, Uvicorn, Pydantic
- **Dashboard:** Streamlit, Plotly

---

## Project Structure
```
customer-segmentation/
├── api/
│   ├── main.py              # FastAPI server and prediction endpoint
│   └── schemas.py           # Pydantic models for API
├── app/
│   └── streamlit_app.py     # Streamlit frontend dashboard
├── data/
│   ├── online_retail_II.csv
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── churn_model.pkl      # Saved XGBoost model and encoders
├── notebooks/
│   └── 01_eda.ipynb         # Initial Exploratory Data Analysis
├── src/
│   ├── data_loader.py       # Data loading utilities
│   ├── feature_engineering.py # RFM calculation and feature encoding
│   └── train.py             # Model training script
├── tests/                   # Pytest test suite
├── Dockerfile               # Container definition
├── render.yaml              # Render deployment configuration
└── requirements.txt         # Project dependencies
```

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MuhammadMinhaj229/customer-segmentation.git
   cd customer-segmentation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the API backend:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 &
   ```

4. **Start the Streamlit dashboard:**
   ```bash
   streamlit run app/streamlit_app.py --server.port=8501
   ```

---

## Limitations & Future Work

- **Toy Datasets:** This project utilizes static, public Kaggle-style datasets (IBM Telco Churn and Online Retail). It does not reflect a live production environment with real-time data ingestion.
- **Model Tuning:** The current model performance (e.g., F1 Score of ~0.54) indicates significant room for improvement through hyperparameter tuning and advanced feature engineering.
- **Incomplete Artifacts:** Some originally planned components (like comprehensive Jupyter notebooks for modeling steps) are not currently present in the repository.
- **Missing Explainability:** Model explainability tools like SHAP have not yet been implemented in the pipeline.

---

## Author / Links
*Built by Mohammed Minhaj Mahmood*
[LinkedIn](https://linkedin.com/in/muhammadminhaj229) · [GitHub](https://github.com/MuhammadMinhaj229)

---

## UNVERIFIED CLAIMS — NOT INCLUDED

The following elements were present in an earlier version of this documentation but have been removed as they could not be verified in the current codebase:
- Claims of "₹1.2–2.4 Crores annual losses", "₹14–18 Lakhs/month revenue saved", and "3.8x Campaign ROI".
- Speculative market-sizing tables ("Industries Where This System Applies").
- Mentions of hyperparameter tuning via GridSearchCV.
- Mentions of model explainability via SHAP.
- Mentions of experiment tracking via MLflow.
- References to missing Jupyter notebooks (`02_rfm_analysis.ipynb`, `03_kmeans_clustering.ipynb`, `04_churn_model.ipynb`).
- Previously inflated performance metrics (AUC 0.89, Silhouette score 0.61, Accuracy 82.3%, F1 0.82).
