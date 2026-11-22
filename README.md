
# 🚗 Used Car Price Prediction

A Machine Learning project that predicts the estimated resale price of used cars based on vehicle specifications and historical data.

## 📌 Project Overview

This project uses Machine Learning regression algorithms to estimate used car selling prices.

The final model is a Random Forest Regressor, integrated into a web application built using Streamlit.

## 🎯 Objectives

- Analyze used car market data.
- Perform data preprocessing and feature engineering.
- Explore relationships between vehicle features and selling prices.
- Train and compare multiple regression models.
- Build an interactive price prediction application.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## 🤖 Machine Learning Models

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

## 🏆 Model Performance

| Model | R² Score |
|---|---:|
| Linear Regression | 0.7359 |
| Decision Tree | 0.8705 |
| Random Forest | 0.9308 |
| Gradient Boosting | 0.9132 |

**Final Model: Random Forest Regressor**

## 🌐 Web Application

The Streamlit application allows users to enter vehicle details and receive an estimated selling price in INR.

## 🚀 How to Run Locally

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
Used-Car-Price-Prediction/
│
├── app.py
├── notebook.ipynb
├── requirements.txt
├── feature_engineered_car_data.csv
└── README.md
```

## 📊 Results

Random Forest achieved the best performance among the four evaluated models, with an R² score of approximately 0.9308.

## 👨‍💻 Project

Used Car Price Prediction  
Machine Learning & Web Application Project
