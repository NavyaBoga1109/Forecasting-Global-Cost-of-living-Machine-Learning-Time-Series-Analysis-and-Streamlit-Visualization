# Cost of Living Analysis and Forecasting

This project aims to analyze the **Global Cost of Living** data and forecast future trends using various machine learning models, including regression and time series analysis. We implemented **ARIMA** and **LSTM** models to forecast the cost trends and created an interactive web-based application using **Streamlit** for data visualization and analysis.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data Overview](#data-overview)
4. [Machine Learning Models](#machine-learning-models)
    - [ARIMA Model](#arima-model)
    - [LSTM Model](#lstm-model)
    - [Regression Models](#regression-models)
5. [Feature Importance](#feature-importance)
6. [Web Application](#web-application)
7. [Results](#results)
8. [Conclusions](#conclusions)

---

## Project Overview

The project analyzes cost-of-living data from different cities and countries, including key factors like **Food Cost**, **Housing Cost**, and **Transportation Cost**. After preprocessing and cleansing the data, we use machine learning algorithms to predict future costs and examine the importance of different features.

### Data Sources
- **Global Cost of Living Dataset** (Kaggle): Dataset includes various cost factors such as food, housing, transportation, utilities, and healthcare, for major cities globally. (https://www.kaggle.com/datasets/mvieira101/global-cost-of-living)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NavyaBoga1109/Forecasting-Global-Cost-of-living-Machine-Learning-Time-Series-Analysis-and-Streamlit-Visualization.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the web application:
   ```bash
   streamlit run web.py
   ```

---

## Data Overview

- **Data Size**: 4874 records and 59 columns.
- **Sample Data**:
    | City     | Country  | Food Cost | Housing Cost | Transportation Cost | Total Cost |
    |----------|----------|-----------|--------------|---------------------|------------|
    | Delhi    | India    | 4.90      | 22.04        | 4.28                | 31.22      |
    | Shanghai | China    | 5.59      | 40.51        | 5.59                | 51.69      |
    | Jakarta  | Indonesia| 2.54      | 22.25        | 3.50                | 28.29      |
    | Manila   | Philippines | 3.54   | 27.40        | 3.54                | 34.48      |
    | Seoul    | South Korea | 7.16   | 52.77        | 6.03                | 65.96      |

---

## Machine Learning Models

### ARIMA Model
We used the **ARIMA** model for time series forecasting of the total cost of living over time. The Mean Squared Error (MSE) for the ARIMA model is:
- **MSE**: 1064.41

![arima_forecasting](https://github.com/user-attachments/assets/35624796-f633-46a6-8adc-bfc34606f020)

### LSTM Model
The **LSTM** model was also used to predict future cost trends. The model was trained over 10 epochs, and the final MSE is:
- **MSE**: 1255.17

![lstm_forecasting](https://github.com/user-attachments/assets/40b459a0-907d-4b68-92aa-e2ee9a25f191)

### Regression Models
We compared different regression models for predicting the total cost based on different factors.

- **Linear Regression MSE**: 6.31e-28
- **Ridge Regression MSE**: 1.71e-08
- **Lasso Regression MSE**: 0.259

![mse_comparison](https://github.com/user-attachments/assets/d589c9f1-9602-4848-b732-13c923530f00)

---

## Feature Importance

Using **Random Forest** for feature importance, we identified **Housing Cost** as the most critical factor contributing to the total cost of living, followed by **Food Cost** and **Transportation Cost**.

- **Housing Cost Importance**: 96.8%
- **Food Cost Importance**: 2.8%
- **Transportation Cost Importance**: 0.3%

---

## Web Application

The project includes a web application built using **Streamlit**. The app provides an interactive interface for exploring the dataset, visualizing key trends, and making predictions based on user inputs.

### Features:
1. **Data Overview**: View a sample of the data and summary statistics.
2. **Visualizations**: Interactive charts for comparing cost factors across different cities.
3. **World Map**: Visualize the cost of living in different locations globally.
4. **Machine Learning Analysis**: Access the results of the regression models, ARIMA, and LSTM forecasting.

![Screenshot (98)](https://github.com/user-attachments/assets/2e1f3370-9f34-435b-8884-51320415f4a5)

![Screenshot (100)](https://github.com/user-attachments/assets/b7020c5f-6660-45bf-8d42-7c083e996a91)

---

## Results

### Model Performance:
- **ARIMA MSE**: 1064.41
- **LSTM MSE**: 1255.17
- **Linear Regression MSE**: 6.31e-28
- **Ridge Regression MSE**: 1.71e-08
- **Lasso Regression MSE**: 0.259

---

## Conclusions

- **Housing Cost** plays a dominant role in predicting the overall cost of living.
- **ARIMA** and **LSTM** models provide useful insights but differ in accuracy.
- The regression models offer precise predictions, with **Ridge Regression** performing the best.

---

## How to Use

1. Run the web application via Streamlit.
2. Navigate through the different sections: Data Overview, Visualizations, Machine Learning Analysis, and World Map.
3. Explore insights about global cost trends, feature importance, and machine learning model predictions.

---
