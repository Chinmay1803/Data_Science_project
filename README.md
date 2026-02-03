# 📊 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a customer is likely to churn based on historical behavior data.  
The trained model is deployed as an interactive **Streamlit web application**.

---

## 🚀 Project Overview

Customer churn is a critical business problem where companies lose customers to competitors.  
This project uses machine learning to identify high-risk customers so that businesses can take proactive retention actions.

**Key Highlights**
- Real-world structured dataset
- End-to-end ML pipeline
- Model evaluation using ROC-AUC
- Interactive web app deployment

---

## 🧠 What This Project Covers

- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training using Random Forest  
- Model evaluation with classification metrics  
- Model persistence using Joblib  
- Deployment using Streamlit  

---

## 🗂️ Project Structure

churn_prediction/
│
├── data.csv # Dataset
├── churn_model.py # Model training script
├── churn_analysis.ipynb # EDA & experimentation notebook
├── app.py # Streamlit web application
├── requirements.txt # Project dependencies
└── README.md # Project documentation

## how to run the code 
pip install -r requirements.txt
python churn_model.py
python -m streamlit run app.py
