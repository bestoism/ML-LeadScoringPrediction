# Predictive Lead Scoring for Term Deposits with Explainable AI

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit%20%26%20FastAPI-orange.svg)](https://streamlit.io/)
[![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%20%7C%20XGBoost%20%7C%20SHAP-green.svg)](https://scikit-learn.org/)

This project develops a complete machine learning solution to predict whether a customer will subscribe to a term deposit at a bank. The core focus is not only on building an accurate predictive model but also on ensuring its transparency and interpretability through **Explainable AI (XAI)**, culminating in a fully interactive web application for model analysis and real-time prediction.

This project was developed as part of an Independent Study program, aligning with the initial academic proposal.

---

## 📋 Project Objectives

As outlined in the proposal, the primary goals were:
1.  **Develop an Accurate Classification Model:** To predict the probability of a customer subscribing to a term deposit.
2.  **Implement Model Interpretation (XAI):** To provide transparent explanations for the model's predictions, understanding the "why" behind the "what".
3.  **Design Actionable Insights:** To translate model interpretations into simple, actionable business logic, such as a "Next Best Conversation" strategy for the sales team.
4.  **Deliver a Usable Solution:** To create a tangible asset that could be used by a collaborator or business partner.

---

## 🚀 The Journey: From V1 to a More Robust V2

A key part of this project was an iterative development process, driven by critical analysis and feedback.

### Initial Approach (V1)
The first version of the model was developed following a standard preprocessing pipeline. While this yielded high performance metrics, a deeper analysis revealed several critical flaws that are common in real-world data science projects.

### Critical Review & Key Learnings
Based on a thorough review (inspired by feedback from my lecturer), several key issues were identified in the V1 preprocessing:

*   **Data Leakage:** The `duration` feature (call duration) was a major predictor. However, this information is only known *after* a call is made, making it unsuitable for a pre-campaign predictive model. Its inclusion led to an overly optimistic and unrealistic model performance.
*   **Flawed Feature Engineering:** The `pdays` feature, which indicates days since the last contact, had a special value `999` for new customers. This was incorrectly mapped to `0`, conflating new customers with those contacted on the same day.
*   **Shallow EDA:** The initial Exploratory Data Analysis lacked bivariate analysis, failing to uncover key relationships between features and the target variable.

### The Improved Approach (V2)
To address these issues and build a more honest and deployable model, the entire preprocessing and modeling pipeline was rebuilt:

1.  **Eliminated Data Leakage:** The `duration` feature was completely removed to ensure the model is truly predictive.
2.  **Corrected Feature Engineering:** The `pdays` column was transformed into a new binary feature `pernah_dihubungi` (was_contacted), accurately capturing this crucial piece of information.
3.  **In-depth EDA:** Bivariate analysis was performed to better understand the data and guide the modeling process.

---

## 🤖 Modeling & Results (V2)

With a robust and clean dataset, several models were trained and evaluated. The final goal was to find the model with the best balance between identifying potential customers (**Recall**) and not wasting the sales team's time (**Precision**), as measured by the **F1-Score**.

### Model Comparison (on the corrected V2 dataset)

| Model                 | Accuracy | Precision (for 'Yes') | Recall (for 'Yes') | F1-Score (for 'Yes') |
| --------------------- | :------: | :-------------------: | :----------------: | :------------------: |
| Logistic Regression   |  0.8454  |         0.3582        |       0.6401       |        0.4593        |
| Random Forest         |  0.8936  |         0.5746        |       0.3341       |        0.4223        |
| XGBoost (Default)     |  0.8904  |         0.5284        |       0.5603       |        0.5439        |
| **XGBoost (Tuned)**   |  **0.8978**  |         **0.6015**        |       **0.4892**       |        **0.5397**        |

*(Note: These are example scores. Replace them with the final scores from your `02B-Modeling.ipynb` summary table.)*

**Conclusion:** The **Hyperparameter-Tuned XGBoost** model was selected as the final model. Although Random Forest has higher precision, the tuned XGBoost provides a superior balance (F1-Score) and a stronger ability to identify potential leads (Recall), which is crucial for maximizing sales opportunities.

---

## 💡 Explainable AI (XAI) with SHAP

To understand the final model's decisions, SHAP (SHapley Additive exPlanations) was implemented.

### Key Feature Importance (V2 Model)

This plot shows the features that have the most impact on the model's predictions, free from data leakage.

*To make this work, save your SHAP bar plot from notebook `02B` into the `reports` folder as `shap_summary_v2.png`*
![SHAP Summary Plot](reports/shap_summary_v2.png)

**Key Insights:**
*   **Economic Indicators are Crucial:** Features like `nr.employed` (number of employees) and `euribor3m` (interest rates) are now top predictors, showing that the customer's decision is heavily influenced by the macroeconomic climate.
*   **Contact History Matters:** The `poutcome_success` feature is highly influential, indicating that customers who have converted in previous campaigns are extremely valuable leads.
*   **Contact Method:** The `contact_cellular` feature is also significant, suggesting the communication channel plays a role in conversion rates.

---

## 🚀 Going Beyond: The Interactive Dashboard

To fulfill the objective of delivering a usable solution and to demonstrate the project's full potential, I went beyond the core ML role to build a full-stack interactive web application.

*To make this work, record a short GIF of you using your Streamlit app and save it in the `reports` folder as `dashboard_demo.gif`*
![Dashboard Demo](reports/dashboard_demo.gif)

This dashboard, built with **Streamlit** (frontend) and **FastAPI** (backend), allows a user to:
*   **Compare All Trained Models:** Select any of the four final models (Logistic Regression, Random Forest, XGBoost Default, XGBoost Tuned) from a dropdown menu.
*   **View Performance Metrics:** Instantly see the Accuracy, Precision, Recall, and F1-Score for the selected model.
*   **Perform Real-Time Predictions:** Input hypothetical customer data using interactive sliders and see the model's prediction and probability score change in real-time.
*   **Get Contextual Analysis:** Read a summary of the selected model's strengths and weaknesses.

---

## 📂 Project Structure

ML-LEADSCORINGPREDICTION/
├── data/
│ ├── bank-additional-full.csv (Raw Data)
│ └── bank_cleaned_v2.csv (Cleaned & Processed Data)
├── models/
│ ├── models_V1/ (Models with data leakage)
│ └── models_V2/ (Final, corrected models)
│ ├── logistic_regression_v2.pkl
│ ├── random_forest_v2.pkl
│ ├── xgboost_default_v2.pkl
│ └── xgboost_tuned_v2.pkl
├── notebooks/
│ ├── 01-EDA.ipynb (Initial exploration)
│ ├── 01B-EDA-Advanced.ipynb (Corrected, in-depth EDA)
│ ├── 02-Modeling.ipynb (Initial modeling)
│ └── 02B-Modeling.ipynb (Final, corrected modeling)
├── reports/ (For storing plots and GIFs)
├── src/
│ └── main.py (FastAPI Backend API)
├── .gitignore
├── app.py (Streamlit Frontend App)
├── README.md
└── requirements.txt

---

## 🛠️ How to Run the Project

Follow these steps to run the interactive web application locally.

**1. Clone the Repository**

git clone <your-repo-url>
cd ML-LEADSCORINGPREDICTION

**2. Create and Activate a Virtual Environment**
# Create venv
python -m venv venv

# Activate venv
# Windows
venv\Scripts\activate
# MacOS/Linux
source venv/bin/activate

**3. Install Dependencies**
pip install -r requirements.txt

**4. Run the Backend API (FastAPI)**
Open a new terminal, navigate to the src directory, and run:

cd src
uvicorn main:app --reload

The API will be running at http://127.0.0.1:8000. Keep this terminal open.

**5. Run the Frontend Application (Streamlit)**
Open a second terminal, navigate to the project's root directory, and run:

streamlit run app.py

A new tab will open in your browser with the interactive dashboard at http://localhost:8501.

💻 Technologies Used
Programming Language: Python
Data Analysis: Pandas, NumPy
Data Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn, XGBoost
Model Interpretation: SHAP
Backend API: FastAPI, Uvicorn
Frontend Web App: Streamlit
Development: Jupyter Notebook, VS Code
