# Credit_Card_Approval-Prediction-
Developed a credit card approval prediction model using demographic and financial data. Performed EDA, feature engineering, and applied ML models (Logistic Regression, Random Forest, XGBoost) to assess credit risk, improving approval accuracy and identifying high-risk applicants.

# Project Overview
This project focuses on predicting whether a **credit card application** will be approved or denied based on applicant demographic and financial features.  
The goal is to assist financial institutions and credit analysts in assessing **creditworthiness**, minimizing **default risk**, and improving **approval decision accuracy**.

##  Problem Statement
Given customer details such as income, employment duration, and credit history, predict whether a credit card application should be **approved (1)** or **denied (0)**.

---
## Dataset
- **Source:** [Kaggle - Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data)
- **Key Features:**
  - Demographics: Age, Gender, Family Size
  - Financial: Income, Employment Duration, Credit History
  - **Target Variable:** Application Status (Approved / Denied)

##  Exploratory Data Analysis (EDA)
- Identified **strong positive correlation** between `children_count` and `family_members` (0.89) → potential redundancy.  
- Found **negative correlation** between `days_birth` and `days_employed` (–0.62) → older applicants have longer employment stability.  
- Used **UMAP visualisation** to detect data complexity and absence of clear clusters, suggesting heterogeneous applicant behavior.

##  Data Preprocessing
- Label Encoding for categorical features  
- Feature scaling using `StandardScaler`  
- Applied **SMOTE** to address class imbalance  
- Created binary label:  
  - `0` → Overdue / No active loan  
  - `1` → Paid-off / Reliable  


##  Machine Learning Models
| Model | Description | Key Techniques |
|--------|--------------|----------------|
| **Logistic Regression** | Baseline model using `RandomizedSearchCV` for C, solver, and iteration tuning | StandardScaler + SMOTE |
| **Random Forest Classifier** | Ensemble model with `GridSearchCV` for depth, estimators, and split criteria | 3-fold CV |
| **Decision Tree Classifier** | Tuned for `max_depth`, `criterion`, and `min_samples_split` to reduce overfitting | ROC-AUC optimization |
| **XGBoost** | Gradient boosting model tuned for `learning_rate`, `n_estimators`, and `subsample` | Binary setup |
| **LightGBM** | Gradient boosting variant optimized for speed and accuracy | Used clustering integration |
| **DBSCAN** | Unsupervised clustering for anomaly detection | Tuned `eps` and `min_samples` |

##  Model Evaluation
- **Random Forest** and **Decision Tree** achieved top accuracy and reliability.  
- **XGBoost** balanced precision and recall, ideal for regulatory risk modeling.  
- **DBSCAN** effectively flagged outliers → potential **fraudulent** or **irregular** applicants.  
- **LightGBM** offered scalability and performance for real-world deployment.
  
## Key Insights
- Combining **predictive modeling** (Random Forest, Decision Tree) with **anomaly detection** (DBSCAN) enhances credit-risk visibility.  
- Feature engineering and data balancing significantly improved recall for minority (risky) applicants.  
- Future iterations could integrate **real-time scoring pipelines** or **automated risk dashboards** for analysts.

##  Tech Stack
- **Languages:** Python  
- **Machine Learning Algorithim & Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, Matplotlib, Seaborn, UMAP  
- **Tools:** Jupyter Notebook, PowerPoint for presentation  

## 👨‍💼 Business Impact
This model provides a **data-driven decision framework** for credit analysts:
- Reduces manual workload in application screening  
- Improves risk prediction accuracy  
- Supports better policy and credit-limit decisions  
---

## Learning Outcomes & Experience Gained

- Gained hands-on experience in **credit risk modeling** and **financial data interpretation**, improving understanding of customer behavior and loan repayment trends.  
- Strengthened expertise in **data preprocessing, EDA, and statistical feature analysis** for decision-making.  
- Learned to handle **imbalanced datasets** using SMOTE and perform **hyperparameter tuning** with GridSearchCV and RandomizedSearchCV.  
- Improved ability to evaluate **model performance metrics** (Accuracy, Precision, Recall, ROC-AUC) for business applicability.  
- Developed teamwork, presentation, and reporting skills through collaborative work and result visualization.  
- Enhanced domain understanding of **credit approval processes, financial indicators, and risk assessment frameworks** used in banking and lending industries.


