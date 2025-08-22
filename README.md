# Customer Intelligence Platform

A full analytics platform integrating **EDA, Customer Segmentation, Churn Prediction, Sales Forecasting, Text Analysis**, and **API Deployment**.

## Features
- **Data Cleaning & EDA**
- **Customer Segmentation (KMeans)**
- **Churn Prediction (ML Classifiers)**
- **Sales Forecasting (Ridge, Linear Regression, etc.)**
- **Text Analysis (Sentiment using NLTK VADER)**
- **API Backend** (Flask + Gunicorn + Docker)
- **Interactive Dashboards** (Power BI)
- **Deployment** (GCP Cloud Run via Docker + GitHub Actions)

## Endpoints
Base URL: `http://localhost:8501` (local) or your Cloud Run URL

- `GET /health` → API health check
- `POST /train` → Train churn model, returns metrics
- `POST /predict/churn` → Predict churn for customers
- `GET /forecast/sales?horizon=6` → Sales forecast
- `POST /sentiment` → Sentiment analysis of feedback text
- `POST /segment/kmeans` → Customer segmentation
- `POST /prescriptions` → Prescriptive analytics (recommendations)

## Run locally with Docker
```bash
docker build -t ci-platform-api:local .
docker run -p 8501:8501 ci-platform-api:local
