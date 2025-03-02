import pandas as pd
import numpy as np
import csv
from pvlib import solarposition

# FILENAMES  ----------------------------------------

input_file = './data/data.xlsx'
output_file = './data/training_data.csv'

# VALUES FOR SOLAR POSITION CALCULATION --------------

lat, lon = 51.1, 17.0  # Wroclaw coordinates

# READ AND PREP EXCEL --------------------------------

df = pd.read_excel(input_file, 'wszystko godzinowe', header=None)

# delete empty rows
df = df.dropna(axis=0, how="all")
# set time format
df[2] = pd.to_datetime(df[2], errors='coerce')
df[2] = df[2].dt.time
df[1] = pd.to_datetime(df[1], errors='coerce')

# setting datetime creates NaT values at midnight so replace with 00:00:00
if df[2].isna().any():
    print("Rows with NaT values:\n", df[df[2].isna()])
    df[2].fillna(pd.to_datetime('00:00:00'), inplace = True)
    print(df.head(10))
    
if df[1].isna().any():
    print("Rows with NaT values:\n", df[df[2].isna()])
    df[1].fillna(pd.to_datetime('2024-07-04'), inplace = True)
    print(df.head(10))

# wxtract row metadata
categories = df.iloc[0, :]
facade_names = df.iloc[2, :]

# weather station columns
weather_data_indices = [7, 9, 13, 14, 15, 16]

# building characteristics
building_info = {
    "C21": {"area": "green", "insulation": "uninsulated", "subcols": 8},
    "Olimpia": {"area": "green", "insulation": "uninsulated", "subcols": 8},
    "Przedmieście Oławskie": {"area": "concrete", "insulation": "insulated", "subcols": 7},
    "Zapolskiej godzinowe": {"area": "concrete", "insulation": "uninsulated", "subcols": 8}
}

output_data = []

def calculate_multiplier(date, hour, surface_azimuth):
    
    # Assuming azimuth in degrees: 0° = North, 90° = East, 180° = South, 270° = West
    # Surface azimuth can be 0 for North, 90 for East, 180 for South, 270 for West
    
    # Calculate the angle difference between sun's azimuth and surface's azimuth
    angle_diff = (azimuth - surface_azimuth + 360) % 360  # Ensure angle is positive
    
    # Define multiplier based on direction and time
    if surface_azimuth == 90:  # East
        if 6 <= hour < 12:  # Morning (6am to 12pm)
            return 1 + 0.5 * np.cos(np.radians(angle_diff))  # Amplify for east-facing surfaces
        else:
            return 1  # No amplification after 12pm
    elif surface_azimuth == 180:  # South
        if 12 <= hour < 18:  # Afternoon (12pm to 6pm)
            return 1 + 0.5 * np.cos(np.radians(angle_diff))  # Amplify for south-facing surfaces
        else:
            return 1  # No amplification before 12pm or after 6pm
    elif surface_azimuth == 270:  # West
        if 15 <= hour < 18:  # Evening (3pm to 6pm)
            return 1 + 0.5 * np.cos(np.radians(angle_diff))  # Amplify for west-facing surfaces
        else:
            return 1  # No amplification outside evening
    else:
        return 1  # Default multiplier for other orientations (e.g., North-facing)


# PROCESS DATA --------------------------------

# loop through each building
for building, info in building_info.items():
    building_col = df.columns[categories == building].tolist()[0]

    subcols = info["subcols"]
    area = info["area"]
    insulation = info["insulation"]

    building_cols = list(range(building_col, building_col + subcols))  # Get the indices of the subcolumns
    print(f"Collecting data from buildign {building}. Using columns: {building_cols}")

    # loop through each sensor for this building by gettng data from the appropriate column
    for col_idx in building_cols:
        
        facade_orientation = str(facade_names[col_idx])[0]  # first letter (N, S, E, W)
        ground_floor = 0 if int(str(facade_names[col_idx])[1]) == 1 else 1        
        facade_n, facade_s, facade_e, facade_w, facade_azimunth = 0, 0, 0, 0, 0
        if facade_orientation == 'N':
            facade_n = 1 
            facade_azimunth = 0
        if facade_orientation == 'S':
            facade_s = 1
            facade_azimunth = 180
        if facade_orientation == 'E':
            facade_e = 1 
            facade_azimunth = 90
        if facade_orientation == 'W':
            facade_w = 1
            facade_azimunth = 270

        # extract temperature data for this facade
        temperatures = df.iloc[3:, col_idx]

        # combine data with weather and metadata
        for idx, temp in enumerate(temperatures):
            
            # extract weather data for the current row
            weather_data = {
                df.iloc[2, weather_idx]: df.iloc[idx + 3, weather_idx]
                for weather_idx in weather_data_indices
            }
            
            date = df.iloc[idx + 3, 1].date()
            hour = df.iloc[idx + 3, 2].hour
            timestamp = pd.to_datetime(f"{date} {hour}:00:00")
            
            solar_pos = solarposition.get_solarposition(timestamp, lat, lon)
            azimuth = solar_pos['azimuth'].values[0]
            
            multiplier = calculate_multiplier(azimuth, hour, facade_azimunth)
            
            # encode hour with sine and cosine for cyclicality
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            sun_orientation_conv = weather_data['Solar Rad'] * facade_azimunth
            
            # prepare the row data
            row_data = {
                "Date": date,
                "Hour": hour,
                "Hour_Sin": round(hour_sin, 3),
                "Hour_Cos": round(hour_cos, 3),
                **weather_data,
                "SunDirectionConv": sun_orientation_conv,
                "Facade_Orientation_N": int(facade_n),
                "Facade_Orientation_S": int(facade_s),
                "Facade_Orientation_E": int(facade_e),
                "Facade_Orientation_W": int(facade_w),
                "Ground_floor": ground_floor,
                "Upper_floor": 0 if ground_floor == 1 else 1,
                "Area_green": 1 if info["area"] == "green" else 0,
                "Area_concrete": 0 if info["area"] == "green" else 1,
                "Insulation_insulated": 1 if info["insulation"] == "insulated" else 0,
                "Insulation_uninsulated": 0 if info["insulation"] == "insulated" else 1,
                "Temperature": round(temp, 1),
            }
            output_data.append(row_data)
         
print("Data processing complete")   
print(f"Dataset size: {len(output_data)}")

# SAVE DATA --------------------------------

fieldnames = list(output_data[0].keys())


with open(output_file, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_data)

print(f"Processed data saved to {output_file}")