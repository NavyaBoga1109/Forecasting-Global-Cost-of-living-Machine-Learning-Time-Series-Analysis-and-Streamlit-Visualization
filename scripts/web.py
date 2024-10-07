import streamlit as st
import pandas as pd
from load_data import load_data
from world import create_world_map
import plotly.express as px
from analysis import regression_comparison, feature_importance, arima_forecasting, lstm_forecasting

# Load data
file_path = 'cleaned_cost_of_living.csv'
df = load_data(file_path)

# Sidebar menu
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Select an option:", ["Data Overview", "Visualizations", "World Map", "Machine Learning Analysis"])

# Data Overview
if menu == "Data Overview":
    st.title("Cost of Living Data Overview")
    st.write("### Data Sample")
    st.dataframe(df.head())
    st.write("### Data Summary")
    st.write(df.describe())

# Visualizations
elif menu == "Visualizations":
    st.title("Cost of Living Analysis")
    
    st.subheader("Top 10 Cities by Housing Cost")
    top_10_housing = df[['city', 'Housing_Cost']].nlargest(10, 'Housing_Cost')
    fig_housing = px.bar(top_10_housing, x='Housing_Cost', y='city', orientation='h', title='Top 10 Cities by Housing Cost')
    st.plotly_chart(fig_housing)

    st.subheader("Top 5 Cities by Total Cost")
    top_5_total_cost = df[['city', 'Total_Cost']].nlargest(5, 'Total_Cost')
    fig_total = px.pie(top_5_total_cost, values='Total_Cost', names='city', title='Top 5 Cities by Total Cost')
    st.plotly_chart(fig_total)

    st.subheader("Distribution of Housing Costs")
    fig_distribution_housing = px.box(df, x='Housing_Cost', title='Distribution of Housing Costs Across Cities')
    st.plotly_chart(fig_distribution_housing)

    st.subheader("Distribution of Food Costs")
    fig_distribution_food = px.histogram(df, x='Food_Cost', nbins=30, title='Distribution of Food Costs Across Cities')
    st.plotly_chart(fig_distribution_food)

# World Map
elif menu == "World Map":
    st.title("Interactive World Map")

    # Call the create_world_map function from world.py
    fig_geo = create_world_map()

    # Display the world map
    st.plotly_chart(fig_geo)

# Machine Learning Analysis
elif menu == "Machine Learning Analysis":
    st.title("Machine Learning Analysis")

    st.subheader("Regression Model Comparison")
    st.write("Evaluating different regression algorithms to predict cost-of-living metrics...")
    regression_comparison(df)
    st.image('regression_comparison.png', caption='Regression Model Comparison')

    st.subheader("Feature Importance")
    st.write("Determining significant factors affecting the cost of living...")
    feature_importance(df)

    st.subheader("ARIMA Forecasting")
    st.write("Forecasting future trends using ARIMA...")
    arima_mse = arima_forecasting(df)
    st.write(f"ARIMA MSE: {arima_mse}")
    st.image('arima_forecasting.png', caption='ARIMA Forecasting Results')

    st.subheader("LSTM Forecasting")
    st.write("Forecasting future trends using LSTM...")
    lstm_mse = lstm_forecasting(df)
    st.write(f"LSTM MSE: {lstm_mse}")
    st.image('lstm_forecasting.png', caption='LSTM Forecasting Results')

    st.subheader("MSE Comparison")
    st.write("Comparing MSE of ARIMA and LSTM...")
    st.image('mse_comparison.png', caption='MSE Comparison: ARIMA vs LSTM')
