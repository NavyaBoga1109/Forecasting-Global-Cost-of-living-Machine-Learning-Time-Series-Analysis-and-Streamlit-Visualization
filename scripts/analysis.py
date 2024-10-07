import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Load the cleaned dataset
df = pd.read_csv('cleaned_cost_of_living.csv')

# Calculate Total_Cost
df['Total_Cost'] = df[['Food_Cost', 'Housing_Cost', 'Transportation_Cost']].sum(axis=1)

# Regression Model Comparison
def regression_comparison(df):
    features = df[['Food_Cost', 'Housing_Cost', 'Transportation_Cost']]
    target = df['Total_Cost']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso()
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        metrics[name] = mse
        print(f"{name} MSE: {mse}")

    # Plotting the results
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green'])
    plt.title('Regression Model Comparison')
    plt.ylabel('Mean Squared Error')
    plt.savefig('regression_comparison.png')  # Save the plot as an image
    plt.close()

# Feature Importance using Random Forest
def feature_importance(df):
    features = df[['Food_Cost', 'Housing_Cost', 'Transportation_Cost']]
    target = df['Total_Cost']
    
    rf_model = RandomForestRegressor()
    rf_model.fit(features, target)
    importance = rf_model.feature_importances_

    for feature, imp in zip(features.columns, importance):
        print(f"Feature: {feature}, Importance: {imp}")

# ARIMA Forecasting
def arima_forecasting(df):
    df_limited = df.head(100).copy()
    df_limited['date'] = pd.date_range(start='1/1/2020', periods=len(df_limited), freq='MS')
    df_limited.set_index('date', inplace=True)

    time_series_data = df_limited['Total_Cost']
    arima_model = sm.tsa.ARIMA(time_series_data, order=(1, 1, 1))
    arima_result = arima_model.fit()

    # Forecast for the next 12 months
    forecast = arima_result.forecast(steps=12)
    
    # Calculate MSE
    actual_values = time_series_data[-12:].values
    arima_mse = mean_squared_error(actual_values, forecast.values)
    print(f"ARIMA MSE: {arima_mse}")

    # Plot actual vs forecasted
    plt.figure(figsize=(10, 6))
    plt.plot(time_series_data, label='Actual Total_Cost')
    plt.plot(forecast.index, forecast, label='Forecasted Total_Cost', linestyle='--', color='red')
    plt.title('ARIMA Model: Actual vs Forecasted Total_Cost')
    plt.legend()
    plt.savefig('arima_forecasting.png')  # Save the plot as an image
    plt.close()

    return arima_mse

# LSTM Forecasting
def lstm_forecasting(df):
    df_limited = df.head(100).copy()
    df_limited['date'] = pd.date_range(start='1/1/2020', periods=len(df_limited), freq='MS')
    df_limited.set_index('date', inplace=True)

    time_series_data = df_limited['Total_Cost'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(time_series_data)

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_dataset(dataset, time_step=10):
        X, y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform to get actual values
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Prepare the actual values for MSE calculation
    actual_test_values = scaler.inverse_transform(test_data[time_step:])

    # Calculate MSE
    lstm_mse = mean_squared_error(actual_test_values, test_predict)
    print(f"LSTM MSE: {lstm_mse}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_series_data, label="Original Total_Cost", color='blue')
    plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict, label="Train Predictions", color='green')
    plt.plot(np.arange(len(train_predict) + (time_step * 2), len(train_predict) + (time_step * 2) + len(test_predict)), test_predict, label="Test Predictions", color='red')
    plt.title('LSTM Model: Actual vs Predicted Total_Cost')
    plt.xlabel('Time')
    plt.ylabel('Total_Cost')
    plt.legend()
    plt.savefig('lstm_forecasting.png')  # Save the plot as an image
    plt.close()

    return lstm_mse

# Run the analyses
arima_mse = arima_forecasting(df)
lstm_mse = lstm_forecasting(df)

# Compare MSEs
plt.figure(figsize=(10, 6))
plt.bar(['ARIMA', 'LSTM'], [arima_mse, lstm_mse], color=['orange', 'green'])
plt.title('MSE Comparison: ARIMA vs LSTM')
plt.ylabel('Mean Squared Error')
plt.savefig('mse_comparison.png')  # Save the plot as an image
plt.close()

# Run additional analysis functions
regression_comparison(df)
feature_importance(df)
