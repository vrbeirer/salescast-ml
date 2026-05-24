# 🚀 SalesCast — Multi-Store Sales Forecasting System

> A full-stack machine learning system that predicts future retail sales using advanced time-series feature engineering and XGBoost.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-success?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![Frontend](https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JS-orange?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-Time_Series-blue?style=for-the-badge&logo=kaggle)

---

# 📌 Overview

SalesCast is an end-to-end machine learning forecasting platform built on the Kaggle competition dataset:

### 📊 Predict Future Sales — 1C Company

The system predicts monthly product sales for thousands of retail products across multiple stores using historical transactional data and advanced time-series feature engineering.

It demonstrates the complete ML lifecycle:

- Data cleaning
- Exploratory analysis
- Feature engineering
- Model training
- Model serialisation
- REST API deployment
- Interactive frontend integration

---

# ✨ Key Highlights

- 🔥 Forecasts monthly sales for **22,000+ items**
- 🏪 Supports predictions across **60 retail stores**
- 📈 Trained on **2.9 million+ sales records**
- ⚡ Built using **XGBoost Regressor**
- 🌐 Full-stack deployment with **Flask + Vanilla JS**
- 🧠 Advanced lag & rolling mean feature engineering
- 🎯 Achieved validation **RMSE ≈ 0.96**

---

# 🖼️ Project Architecture

```text
Raw Sales Data
       ↓
Data Cleaning & EDA
       ↓
Feature Engineering
       ↓
XGBoost Training
       ↓
Model Serialization (.pkl)
       ↓
Flask REST API
       ↓
Interactive Web Frontend
```

---

# 🧠 Machine Learning Pipeline

## 1️⃣ Data Cleaning

The raw dataset contained noisy and inconsistent data.

### Cleaning Steps

- Removed extreme outliers
- Merged duplicate shop names using fuzzy matching
- Fixed inconsistent category naming
- Parsed item/category text fields
- Removed invalid sales entries

### Techniques Used

```python
FuzzyWuzzy
Regex Parsing
Outlier Removal
Label Encoding
```

---

## 2️⃣ Feature Engineering

This project heavily focuses on **time-series feature engineering**.

### Lag Features

Previous sales history was converted into predictive signals.

```python
item_cnt_month_lag_1
item_cnt_month_lag_2
item_cnt_month_lag_3
```

### Rolling Mean Features

Mean encodings were generated across multiple dimensions:

- Item-level trends
- Shop-level trends
- Shop-item interactions
- Category-based behaviour
- Price-based patterns

### Text Features

Item names were parsed to extract:

- Platform names
- Product types
- Sub-categories

---

## 3️⃣ Model Training

Two models were tested:

| Model | Validation RMSE |
|---|---|
| Linear Regression | ~1.20 |
| XGBoost Regressor | **~0.96** |

---

# ⚙️ XGBoost Configuration

```python
XGBRegressor(
    max_depth=10,
    n_estimators=100,
    min_child_weight=0.5,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.1,
    eval_metric="rmse",
    early_stopping_rounds=20
)
```

---

# 🌐 Web Application

The project includes a complete production-style frontend.

## 🏠 Landing Page

Features:

- Hero dashboard section
- Live analytics preview
- Model statistics
- Workflow explanation
- Dataset insights
- Call-to-action sections

---

## 📊 Forecast Tool

Users can:

- Enter `Shop ID`
- Enter `Item ID`
- Generate predictions instantly

### UI Features

- Animated loading screens
- Progress indicators
- Validation handling
- Result cards
- Responsive design

---

# 🔌 REST API

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Home page |
| GET | `/predict.html` | Forecast page |
| POST | `/api/predict` | Generate prediction |

---

## Sample API Request

```json
{
  "shop_id": 5,
  "item_id": 5037
}
```

---

## Sample API Response

```json
{
  "shop_id": 5,
  "item_id": 5037,
  "predicted_sales": 7.42
}
```

---

# 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Framework | XGBoost |
| Data Processing | Pandas, NumPy |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Matplotlib, Seaborn |
| Serialization | Pickle |
| Utilities | FuzzyWuzzy |

---

# 📂 Dataset

### Kaggle Competition

📌 Predict Future Sales

- 2.9M daily sales transactions
- 34 months of historical data
- 22K+ products
- 60 shops

### Files Used

```text
sales_train.csv
items.csv
item_categories.csv
shops.csv
test.csv
```

---

# 🚀 Core ML Concepts Demonstrated

✅ Time-Series Forecasting  
✅ Lag Features  
✅ Rolling Mean Encoding  
✅ Data Leakage Prevention  
✅ Feature Engineering  
✅ Gradient Boosting  
✅ Model Deployment  
✅ REST API Development  
✅ Full-Stack ML Integration  

---

# 📈 Results

| Metric | Score |
|---|---|
| Validation RMSE | **~0.96** |
| Dataset Size | 2.9M+ rows |
| Shops | 60 |
| Products | 22,170+ |

---

# 💡 What Makes This Project Strong?

This is not just a notebook model.

It demonstrates:

- Real-world ML engineering
- Production deployment workflow
- Full-stack integration
- Time-series understanding
- Feature engineering depth
- API-based inference serving

---

# 🧪 Future Improvements

- Docker deployment
- Cloud hosting (AWS/GCP)
- Real-time streaming predictions
- Automated retraining pipeline
- Dashboard analytics
- LightGBM/CatBoost experimentation

---

# 👨‍💻 Author

**Pranav**  
Machine Learning Engineer | AI Engineer

---

# ⭐ If You Like This Project

Give it a ⭐ on GitHub and feel free to fork it.
