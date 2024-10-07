from opencage.geocoder import OpenCageGeocode
import pandas as pd
import plotly.express as px

# Initialize the geocoder with your OpenCage API key
key = 'a200a6456d204d55ae4a13c9a4246951'
geocoder = OpenCageGeocode(key)

# Function to get latitude and longitude
def get_lat_lon(city):
    results = geocoder.geocode(city)
    if results:
        return results[0]['geometry']['lat'], results[0]['geometry']['lng']
    return None, None

# Function to create the world map
def create_world_map():
    # Load the cleaned dataset
    file_path = 'cleaned_cost_of_living.csv'
    df = pd.read_csv(file_path)

    # Limit the data to the top 50 cities for faster testing
    df = df.head(50)

    # Calculate the Total_Cost by summing up Food, Housing, and Transportation costs
    df['Total_Cost'] = df[['Food_Cost', 'Housing_Cost', 'Transportation_Cost']].sum(axis=1)

    # Apply the function to get lat/lon for all cities
    df['Latitude'], df['Longitude'] = zip(*df['city'].apply(get_lat_lon))

    # Filter out any rows where lat/lon is None (failed lookups)
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Create a scatter_geo plot
    fig = px.scatter_geo(
        df,
        lat='Latitude',
        lon='Longitude',
        hover_name='city',
        hover_data={'Total_Cost': True, 'Housing_Cost': True, 'Food_Cost': True, 'Transportation_Cost': True},
        size='Total_Cost',  # Size of markers based on the Total Cost
        color='Housing_Cost',  # Color of markers based on Housing Cost
        projection="natural earth",
        title="Cost of Living Across Cities",
    )

    return fig
