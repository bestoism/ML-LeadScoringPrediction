# Predictive Lead Scoring for Term Deposit Subscriptions: End-to-End

An end-to-end project that builds a valid, interpretable Machine Learning model to identify potential customers for a bank's term deposit campaign, complete with deployment prototypes using FastAPI and Streamlit. This project was developed as part of an Independent Study program in collaboration with PT Dicoding Akademi Indonesia.

---

## Table of Contents
* [Project Overview](#project-overview)
* [Project Objectives](#project-objectives)
* [Methodology & Workflow](#methodology--workflow)
* [Tech Stack](#tech-stack)
* [File Structure](#file-structure)
* [Setup & Installation](#setup--installation)
* [How to Use the Prototypes](#how-to-use-the-prototypes)
* [Deployment Prototypes (API & Dashboard)](#deployment-prototypes-api--dashboard)
* [Final Model Performance (V2 - Validated)](#final-model-performance-v2---validated)
* [Key Findings & Business Insights (XAI)](#key-findings--business-insights-xai)
* [Challenges & Key Learnings](#challenges--key-learnings)
* [Future Improvements](#future-improvements)

---

## Project Overview

In a competitive banking industry, optimizing marketing efforts is crucial. This project addresses this challenge by developing a machine learning model that predicts the likelihood of a customer subscribing to a term deposit.

The primary goal is not just to create an accurate prediction, but to build a **transparent, valid, and interpretable model**. The project journey documents a critical transition from a flawed initial model (V1) to a robust, business-ready final model (V2). To showcase the end-to-end vision, this repository also includes deployment prototypes: a **FastAPI backend** for serving predictions and a **Streamlit dashboard** for interactive analysis.

---

## Project Objectives

1.  **Develop an Accurate Classification Model:** To predict the probability of a prospect converting based on **pre-call information only**.
2.  **Implement Model Interpretation:** To provide transparent explanations for the model's predictions using XAI techniques like SHAP.
3.  **Design Actionable Insights:** To translate model interpretations into simple, actionable insights for the sales team.
4.  **Prototype a Deployed Solution:** To demonstrate how the final model can be integrated into a real-world application using a REST API and an interactive dashboard.

---

## Methodology & Workflow

This project followed a rigorous, iterative lifecycle, which was crucial in overcoming initial data challenges.

1.  **Initial Exploration & Modeling (V1):** The project began with a standard EDA and modeling process, which produced models with deceptively high performance metrics (F1-Score ~0.64).
2.  **Critical Analysis & Identification of Data Leakage:** A deeper review revealed a critical flaw: the `duration` feature (call duration) was a "leaky" predictor. This made the V1 models invalid for pre-call prediction.
3.  **Revised Data Analysis (EDA V2):** A completely new, more profound EDA was conducted, which involved removing the leaky feature, performing in-depth bivariate analysis, and executing smart feature engineering.
4.  **Robust Modeling (V2):** The models were retrained on the valid, non-leaky dataset, with a focus on business-relevant metrics (F1-Score, Precision, Recall).
5.  **Model Tuning & Interpretation:** The best-performing model (XGBoost V2) was fine-tuned and interpreted with SHAP to extract meaningful business insights.
6.  **Prototyping & Deployment:** To demonstrate the project's practical application, two prototypes were developed:
    *   A **FastAPI backend** (`src/main.py`) to serve model predictions via a REST API.
    *   A **Streamlit frontend** (`app.py`) to create an interactive dashboard for model comparison and real-time prediction.

---

## Tech Stack
*   **Data Science:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP
*   **Data Visualization:** Matplotlib, Seaborn
*   **Backend API:** FastAPI, Uvicorn
*   **Frontend Dashboard:** Streamlit
*   **Environment:** Jupyter Notebooks, Python 3.x

---

## File Structure

```
ML-LEADSCORINGPREDICTION/
│
├── .data/
│   ├── bank-additional-full.csv      # Raw dataset
│   └── bank_additional_cleaned_1B.csv  # Final, valid dataset (V2)
│
├── .models/
│   ├── models_V1/                    # Leaky models (archive, not for use)
│   └── models_V2/                    # Final, valid models for production
│       └── xgboost_tuned_v2.pkl      # The final, chosen model
│
├── .notebooks/
│   ├── 01B-EDA.ipynb                 # V2: In-depth, revised, and valid EDA
│   └── 02B-Modeling.ipynb            # V2: Final, robust modeling and interpretation
│
├── .reports/
│   └── shap_summary_plot_v2.png      # SHAP plot for the final model
│
├── .src/
│   └── main.py                       # FastAPI backend logic
│
├── app.py                            # Streamlit interactive dashboard
├── requirements.txt                  # Required Python libraries
└── README.md                         # This file
```

---

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ML-LEADSCORINGPREDICTION.git
    cd ML-LEADSCORINGPREDICTION
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # Windows
    # source venv/bin/activate  # macOS/Linux
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Use the Prototypes

You can run the backend API and the frontend dashboard independently.

### Running the API (Backend)
Navigate to the root directory and run the following command:
```bash
uvicorn src.main:app --reload
```
The API will be available at `http://127.0.0.1:8000/docs` for interactive documentation.

### Running the Dashboard (Frontend)
In a new terminal, navigate to the root directory and run:
```bash
streamlit run app.py
```
The interactive dashboard will open in your browser.

---

## Deployment Prototypes (API & Dashboard)

This project includes two functional prototypes to demonstrate how the ML model can be deployed.

*   **FastAPI Backend:** Provides a REST API endpoint (`/predict`) that can be called by other services to get predictions in a structured JSON format.
*   **Streamlit Dashboard:** An interactive web application that allows users to select a model, input customer data via sliders, and see the prediction in real-time.

> **IMPORTANT NOTE:** The current versions of `app.py` and `src/main.py` are **proof-of-concept prototypes** built using the initial **V1 models**. They still include the flawed `duration` feature as an input. This was done to quickly build the application structure. To make them production-ready, they **must be refactored** to:
> 1.  Load the final `xgboost_tuned_v2.pkl` model.
> 2.  Remove `duration` from the input fields and API schema.
> 3.  Incorporate the preprocessing steps from the `01B-EDA.ipynb` notebook (e.g., creating the `pernah_dihubungi` feature) within the application logic before making a prediction.

---

## From Flawed V1 to Validated V2: A Project Journey

The journey from the initial models (V1) to the final, validated models (V2) is the most critical story of this project. The V1 models, while appearing powerful on the surface, were built on a flawed foundation. V2 represents a robust, methodologically sound solution that is ready for real-world application.

### Complete Model Performance Comparison

The table below presents the performance metrics for all models developed during this project. It clearly illustrates the deceptive performance of the V1 models (due to data leakage) versus the realistic performance of the valid V2 models.

| Version | Model | Accuracy | Precision (Yes) | Recall (Yes) | F1-Score (Yes) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **V1 (Flawed)** | Logistic Regression | 0.8640 | 0.4500 | **0.9100** | 0.6000 |
| **V1 (Flawed)** | Random Forest | **0.9148** | **0.6900** | 0.4400 | 0.5400 |
| **V1 (Flawed)** | XGBoost (Tuned) | 0.8886 | 0.5000 | 0.8800 | **0.6400** |
| --- | --- | --- | --- | --- | --- |
| **V2 (Valid)** | Logistic Regression | 0.8309 | 0.3602 | 0.6455 | 0.4624 |
| **V2 (Valid)** | Random Forest | 0.8935 | 0.5537 | 0.2834 | 0.3749 |
| **V2 (Valid)** | XGBoost (Default) | 0.8451 | 0.3816 | 0.6045 | 0.4679 |
| **V2 (Valid)** | **XGBoost (Tuned)** | **0.8537** | **0.4064** | **0.6476** | **0.4994** |

### Why V2 is Superior: EDA and Modeling Improvements

The V2 iteration is fundamentally better because it addresses critical flaws discovered in the V1 process. The improvements were made across both the analysis and modeling phases.

#### **1. Improvements in Exploratory Data Analysis (EDA)**

*   **Data Leakage Elimination:** The single most important improvement was the **removal of the `duration` feature**. V1 used call duration as a predictor, which is information only available *after* a call is finished. This "leaked" information from the future into the model, making its predictions invalid for pre-call targeting. V2 was built exclusively on pre-call information, ensuring its real-world validity.
*   **From Surface-Level to Deep Analysis:** EDA V1 was limited to basic, univariate analysis. EDA V2 implemented **rich bivariate analysis**, using visualizations to explore the relationship between each feature and the campaign's outcome. This generated crucial business hypotheses *before* modeling even began.
*   **Intelligent Feature Engineering:** The ambiguous `pdays` column (where `999` meant "not previously contacted") was incorrectly handled in V1. In V2, it was transformed into a clear, informative binary feature called `pernah_dihubungi` (was_contacted), which proved to be a significant predictor.

#### **2. Improvements in Modeling & Evaluation**

*   **Honest and Realistic Performance:** The V1 models showed inflated metrics because they were "cheating" with the leaky `duration` feature. The V2 models, trained on a valid dataset, show **honest and reliable performance metrics**. A lower but truthful F1-Score of ~0.50 is infinitely more valuable to a business than a deceptive score of 0.64 that cannot be replicated.
*   **Meaningful Interpretability (XAI):** The SHAP analysis on V1 was useless, as it was completely dominated by the `duration` feature. The SHAP analysis on V2, however, reveals the **true drivers of customer conversion**, such as macroeconomic conditions (`nr.employed`), contact methods (`contact_telephone`), and past campaign outcomes (`poutcome_success`). This provides actionable insights, fulfilling a core objective of the proposal.

### Why XGBoost (Tuned) V2 is the Final Choice

Among the valid V2 models, the **Tuned XGBoost model** was definitively selected for three key reasons:

1.  **Best Performance on the Right Metric:** It achieved the **highest F1-Score (0.4994)**. For this business problem, the F1-Score is the most crucial metric as it measures the harmonic balance between Precision (not wasting the sales team's time) and Recall (not missing out on potential customers).
2.  **Proven Interpretability:** We successfully applied SHAP to this model, generating clear, actionable insights that directly address the "Explainable AI" goal of the project. We know not only *what* it predicts, but *why*.
3.  **Robustness and Industry Standard:** XGBoost is a powerful and widely-used algorithm in the industry, known for its performance and efficiency. This makes it a reliable and scalable choice for a production environment.

## Final Model Performance (V2 - Validated)

The following table summarizes the performance of the **final, valid models (V2)**. These metrics are realistic and reflect the model's true pre-campaign predictive power.

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1-Score (Yes) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **0.8537** | **0.4064** | **0.6476** | **0.4994** |

The **Tuned XGBoost model (V2)** was selected as the final model due to its superior **F1-Score**, representing the best balance between business objectives.

---
