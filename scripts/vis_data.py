import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import streamlit as st

# Load data function
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Rename columns for better clarity
    df.rename(columns={
        'x1': 'Food_Cost',
        'x2': 'Housing_Cost',
        'x3': 'Transportation_Cost',
        'x4': 'Utilities_Cost',
        'x5': 'Healthcare_Cost',
        'x6': 'Education_Cost',
        'x7': 'Entertainment_Cost',
        'x8': 'Clothing_Cost',
        'x9': 'Grocery_Cost',
        'x10': 'Dining_Out_Cost',
        'x11': 'Internet_Cost',
        'x12': 'Phone_Cost',
        'x13': 'Insurance_Cost',
        'x14': 'Miscellaneous_Cost_1',
        'x15': 'Miscellaneous_Cost_2',
        'x16': 'Miscellaneous_Cost_3',
        'x17': 'Miscellaneous_Cost_4',
        'x18': 'Miscellaneous_Cost_5',
        'x19': 'Miscellaneous_Cost_6',
        'x20': 'Miscellaneous_Cost_7',
        'x21': 'Miscellaneous_Cost_8',
        'x22': 'Miscellaneous_Cost_9',
        'x23': 'Miscellaneous_Cost_10',
        'x24': 'Miscellaneous_Cost_11',
        'x25': 'Miscellaneous_Cost_12',
        'x26': 'Miscellaneous_Cost_13',
        'x27': 'Miscellaneous_Cost_14',
        'x28': 'Miscellaneous_Cost_15',
        'x29': 'Miscellaneous_Cost_16',
        'x30': 'Miscellaneous_Cost_17',
        'x31': 'Miscellaneous_Cost_18',
        'x32': 'Miscellaneous_Cost_19',
        'x33': 'Miscellaneous_Cost_20',
        'x34': 'Miscellaneous_Cost_21',
        'x35': 'Miscellaneous_Cost_22',
        'x36': 'Miscellaneous_Cost_23',
        'x37': 'Miscellaneous_Cost_24',
        'x38': 'Miscellaneous_Cost_25',
        'x39': 'Miscellaneous_Cost_26',
        'x40': 'Miscellaneous_Cost_27',
        'x41': 'Miscellaneous_Cost_28',
        'x42': 'Miscellaneous_Cost_29',
        'x43': 'Miscellaneous_Cost_30',
        'x44': 'Miscellaneous_Cost_31',
        'x45': 'Miscellaneous_Cost_32',
        'x46': 'Miscellaneous_Cost_33',
        'x47': 'Miscellaneous_Cost_34',
        'x48': 'Miscellaneous_Cost_35',
        'x49': 'Miscellaneous_Cost_36',
        'x50': 'Miscellaneous_Cost_37',
        'x51': 'Miscellaneous_Cost_38',
        'x52': 'Miscellaneous_Cost_39',
        'x53': 'Miscellaneous_Cost_40',
        'x54': 'Miscellaneous_Cost_41',
        'x55': 'Miscellaneous_Cost_42',
        'x56': 'Data_Quality'
    }, inplace=True)

    # Save cleaned dataset
    df.to_csv('cleaned_cost_of_living.csv', index=False)
    return df

def plot_top_10_housing_costs(df):
    top_10_housing = df[['city', 'Housing_Cost']].nlargest(10, 'Housing_Cost')
    plt.figure(figsize=(10, 6))
    plt.barh(top_10_housing['city'], top_10_housing['Housing_Cost'], color='skyblue')
    plt.xlabel('Housing Cost')
    plt.title('Top 10 Cities by Housing Cost')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt

def plot_top_5_total_costs(df):
    top_5_total_cost = df[['city', 'Total_Cost']].nlargest(5, 'Total_Cost')
    plt.figure(figsize=(8, 8))
    plt.pie(top_5_total_cost['Total_Cost'], labels=top_5_total_cost['city'], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Top 5 Cities by Total Cost')
    plt.tight_layout()
    return plt

def plot_housing_cost_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.boxplot(df['Housing_Cost'].dropna(), vert=False, patch_artist=True)
    plt.xlabel('Housing Cost')
    plt.title('Distribution of Housing Costs Across Cities')
    plt.tight_layout()
    return plt

def plot_food_cost_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['Food_Cost'].dropna(), bins=30, color='green', edgecolor='black')
    plt.xlabel('Food Cost')
    plt.ylabel('Frequency')
    plt.title('Distribution of Food Costs Across Cities')
    plt.tight_layout()
    return plt

def plot_housing_vs_transportation(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Housing_Cost'], df['Transportation_Cost'], alpha=0.6)
    plt.xlabel('Housing Cost')
    plt.ylabel('Transportation Cost')
    plt.title('Housing Cost vs Transportation Cost')
    plt.tight_layout()
    return plt

def plot_top_10_food_costs(df):
    top_10_food = df[['city', 'Food_Cost']].nlargest(10, 'Food_Cost')
    plt.figure(figsize=(10, 6))
    plt.barh(top_10_food['city'], top_10_food['Food_Cost'], color='orange')
    plt.xlabel('Food Cost')
    plt.title('Top 10 Cities by Food Cost')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt

def plot_correlation_heatmap(df):
    corr = df[['Food_Cost', 'Housing_Cost', 'Transportation_Cost']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Between Different Cost Categories')
    plt.tight_layout()
    return plt

def plot_grouped_bar_chart(df):
    top_10_cities = df[['city', 'Food_Cost', 'Housing_Cost', 'Transportation_Cost']].head(10).set_index('city')
    top_10_cities.plot(kind='bar', figsize=(12, 6))
    plt.title('Cost Comparison Across Top 10 Cities')
    plt.ylabel('Cost')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def plot_stacked_bar_chart(df):
    top_10_cities = df[['city', 'Food_Cost', 'Housing_Cost', 'Transportation_Cost']].head(10).set_index('city')
    top_10_cities.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Stacked Cost of Living for Top 10 Cities')
    plt.xlabel('City')
    plt.ylabel('Cost')
    plt.legend(title='Cost Category')
    plt.tight_layout()
    return plt

# Load the cleaned dataset
file_path = 'cleaned_cost_of_living.csv'
df = load_data(file_path)

# Calculate Total_Cost
df['Total_Cost'] = df[['Food_Cost', 'Housing_Cost', 'Transportation_Cost', 'Utilities_Cost', 'Healthcare_Cost', 'Education_Cost', 'Entertainment_Cost', 'Clothing_Cost', 'Grocery_Cost', 'Dining_Out_Cost', 'Internet_Cost', 'Phone_Cost', 'Insurance_Cost']].sum(axis=1)

# Function to plot all visualizations
def plot_all_visualizations(df):
    plots = {}
    plots['top_10_housing'] = plot_top_10_housing_costs(df)
    plots['top_5_total'] = plot_top_5_total_costs(df)
    plots['housing_cost_distribution'] = plot_housing_cost_distribution(df)
    plots['food_cost_distribution'] = plot_food_cost_distribution(df)
    plots['housing_vs_transportation'] = plot_housing_vs_transportation(df)
    plots['top_10_food'] = plot_top_10_food_costs(df)
    plots['correlation_heatmap'] = plot_correlation_heatmap(df)
    
    return plots
