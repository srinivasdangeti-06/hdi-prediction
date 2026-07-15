# 🌍 HDI Predictor - Human Development Index Predictor

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

The **HDI Predictor** is a Machine Learning web application that predicts the Human Development Index (HDI) score of countries based on four key development indicators:

- **Life Expectancy** (years)
- **Mean Years of Schooling**
- **Expected Years of Schooling** 
- **GNI per Capita** (USD PPP)

The application uses **Linear Regression** with an **R² Score of ~0.90** (90% accuracy) and provides an interactive web interface built with **Flask**.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 📊 **HDI Prediction** | Predict HDI scores using 4 key indicators |
| 🌍 **Country Selection** | Select from 30+ countries with preset values |
| 🎯 **Quick Test Values** | Pre-set values for Very High, High, Medium, Low categories |
| 📈 **Data Visualizations** | Distribution plots, heatmaps, scatter plots |
| 🏷️ **Category Classification** | Very High, High, Medium, Low |
| 🔗 **REST API** | Programmatic access for integration |
| 📱 **Responsive Design** | Mobile-friendly interface |

---

## 🛠️ Technology Stack

### Backend
- **Python 3.9+** - Programming language
- **Flask** - Web framework
- **Scikit-learn** - Machine Learning (Linear Regression)
- **Pandas** - Data processing
- **NumPy** - Numerical computing

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **Bootstrap 5** - Responsive design

### Visualization
- **Matplotlib** - Charts and plots
- **Seaborn** - Statistical visualizations

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **R² Score** | ~0.90 (90% accuracy) |
| **RMSE** | ~0.023 (Very low error) |
| **MAE** | ~0.019 (High accuracy) |

### Features Importance

| Feature | Coefficient |
|---------|-------------|
| GNI per Capita | 0.035 |
| Life Expectancy | 0.028 |
| Mean Years of Schooling | 0.022 |
| Expected Years of Schooling | 0.018 |

---


 Usage Guide
Web Interface
Home Page - Introduction to HDI Predictor

Prediction Page - Enter indicators and get HDI score

Visualizations - View data insights and charts

About Page - Project documentation

Quick Test Values
Category	Life Expectancy	Mean Schooling	Expected Schooling	GNI per Capita
Very High	80	14	16	50000
High	72	10	12	25000
Medium	65	7	8	10000
Low	55	3	4	3000
