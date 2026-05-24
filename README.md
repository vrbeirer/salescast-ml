# SalesCast — AI-Powered Sales Forecasting

Predict future retail product sales using machine learning, trained on historical data from 60 stores and 22,000+ items.

**Live Demo:** [salescast.onrender.com](https://salescast.onrender.com)

---

## About

SalesCast is an end-to-end ML system built on the Kaggle "Predict Future Sales" dataset by 1C Company. It forecasts unit sales per shop-item pair for November 2015 using 34 months of transactional history.

## Tech Stack

- **Model:** XGBoost (Validation RMSE ~0.96)
- **Backend:** Python · Flask
- **Frontend:** HTML · CSS · JavaScript
- **Data:** Pandas · NumPy · Scikit-learn

## Features

- Sales forecast for any valid Shop ID + Item ID combination
- Production-grade web interface with animated loading states
- REST API served via Flask + Gunicorn

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`

---

© Designed & Developed by Pranav · 2026
