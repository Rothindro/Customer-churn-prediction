# Predictive Modeling of Customer Churn for ADFC Bank Using Machine Learning.
---
## 🎯Overview:
Customer churn is one of the most critical challenges facing ADFC Bank, directly impacting revenue, profitability, and long-term growth. This project aims to analyze historical banking data to predict customer attrition before it happens by leveraging machine learning. This project seeks to imply data-driven strategy by moving beyond intuition to AI-powered insights on why customers leave—whether due to dissatisfaction, competitive offers, or life-stage changes by identifying at-risk customers early so that ADFC Bank can deploy targeted interventions (e.g., personalized offers, service recovery) to improve loyalty.

Extrcated customer attrition insights dataset from ([Kaggle](https://www.kaggle.com/datasets/marusagar/bank-customer-attrition-insights) using Kaggle API:

- __Bank-Customer-Attrition-Insights-Data__


In this project, we have utilized the above mentioned dataset, which contains contains **10,000 records** of ADFC Bank customers with 18 attributes. Key features include:

| Feature              | Relevance to Churn                               |
|----------------------|--------------------------------------------------|
| **RowNumber**        | Sequential record identifier                     |
| **CustomerId**       | Unique customer identifier                       |
| **Surname**          | Customer's last name                             |
| **CreditScore**      | Numerical assessment of creditworthiness         |
| **Geography**        | Customer's location                              |
| **Gender**           | Customer's sex (Male/Female)                     |
| **Age**              | Customer age                                     |
| **Tenure**           | Years with bank                                  | 
| **Balance**          | Account balance                                  |
| **NumOfProducts**    |: Bank products held                              |
| **HasCrCard**        |Credit card ownership                             |
| **IsActiveMember**   |Account activity status                           |
| **EstimatedSalary**  |Annual income                                     |
| **Complain**         |Complaint history                                 |
| **SatisfactionScore**| Resolution satisfaction                          |
| **CardType**         | Credit card type                                 | 
| **PointsEarned**     |Loyalty points                                    |
| **Exited**           |Target variable (1=churned, 0=retained).          |


## ⚙️Methods
The project employs time series analysis techniques, including:

- __Data Extraction__
- __Data Cleaning & Preprocessing__
- __Feature Engineering__
- __Feature Scaling__
- __Modeling__
- __Pediction and Evaluation__


![Machine Learning Pieline]("Customer churn prediction app/pipeline.png")
  

## 🛠️ Tools & Technologies
- __Programming Language:__ Python
- __Libraries:__ NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn, XGBoost, imbalanced-learn
- __Environment:__ Jupyter Notebook
- __Deployment:__ Streamlit

The project will deliver a robust and scalable classification model capable of predicting customer churn. Further this model can be integrated with an application or deployed as a lone model to predict customer churn.
