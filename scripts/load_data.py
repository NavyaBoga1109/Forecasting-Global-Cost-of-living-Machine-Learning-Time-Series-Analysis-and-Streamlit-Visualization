import pandas as pd

def load_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop the "Unnamed: 0" column (assuming it's just an index)
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    # Handle missing values only in numeric columns: fill missing values with the median of each column
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
        'data_quality': 'Data_Quality'
    }, inplace=True)

    # Calculate Total Cost
    df['Total_Cost'] = df[['Food_Cost', 'Housing_Cost', 'Transportation_Cost']].sum(axis=1)

    # Print the cleaned data for verification
    print("Columns in DataFrame:", df.columns.tolist())
    print(df.head())

    # Save the cleaned dataset for future use
    df.to_csv('cleaned_cost_of_living.csv', index=False)

    return df

# Load the dataset
if __name__ == "__main__":
    file_path = r"C:\Users\navya\Downloads\FDS PROJ\cost-of-living.csv\cost-of-living.csv"
    load_data(file_path)