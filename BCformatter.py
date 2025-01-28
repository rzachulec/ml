import pandas as pd
import numpy as np
import csv

file = './data/data.xlsx'

# Read the spreadsheet
df = pd.read_excel(file, 'wszystko godzinowe', header=None)

df = df.dropna(axis=0, how="all")
df[2] = pd.to_datetime(df[2], errors='coerce')

if df[2].isna().any():
    print("Rows with NaT values:\n", df[df[2].isna()])
    df[2].fillna(pd.to_datetime('00:00:00'), inplace=True)    
    print(df.head(10))

# Load the dataset
output_file = './data/training_data.csv'

# Extract row metadata
categories = df.iloc[0, :]  # First row
facade_names = df.iloc[2, :]  # Third row for facade orientations and floor number

# Weather station columns

weather_data_indices = [7, 9, 13, 14, 15, 16]

# Building name assignments
building_info = {
    "C21": {"area": "green", "insulation": "uninsulated", "subcols": 8},
    "Olimpia": {"area": "green", "insulation": "uninsulated", "subcols": 8},
    "Przedmieście Oławskie": {"area": "concrete", "insulation": "insulated", "subcols": 7},
    "Zapolskiej godzinowe": {"area": "concrete", "insulation": "uninsulated", "subcols": 8}
}

output_data = []


# Process each building category
for building, info in building_info.items():
    building_col = df.columns[categories == building].tolist()[0]

    subcols = info["subcols"]
    area = info["area"]
    insulation = info["insulation"]

    building_cols = list(range(building_col, building_col + subcols))  # Get the indices of the subcolumns
    print(building_cols)

    # Loop through each subcolumn for this building
    for col_idx in building_cols:
        
        facade_orientation = str(facade_names[col_idx])[0]  # First letter (N, S, E, W)
        floor_level = 1 if int(str(facade_names[col_idx])[1]) == 1 else 2

        print(facade_orientation)

            # Extract temperature data for this facade
        temperatures = df.iloc[3:, col_idx]  # Assuming data starts from 4th row (index 3)

            # Combine data with weather and metadata
        for idx, temp in enumerate(temperatures):
            # Extract weather data for the current row
            weather_data = {
                df.iloc[2, weather_idx]: df.iloc[idx + 3, weather_idx]
                for weather_idx in weather_data_indices
            }
            
            # Extract the hour of the day
            hour = df.iloc[idx + 3, 2].hour
            
            # Encode hour with sine and cosine for cyclicality
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            # Prepare the row data with the encoded time
            row_data = {
                "Hour": hour,
                "Hour_Sin": round(hour_sin, 3),
                "Hour_Cos": round(hour_cos, 3),
                **weather_data,
                "Facade_Orientation": facade_orientation,
                "Floor_Level": floor_level,
                "Area": 1 if info["area"] == "green" else 0,
                "Insulation": 1 if info["insulation"] == "insulated" else 0,
                "Temperature": round(temp, 1),
            }
            output_data.append(row_data)

print(len(output_data))

fieldnames = list(output_data[0].keys())


with open(output_file, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Write the header (column names)
    writer.writeheader()
    
    # Write all rows
    writer.writerows(output_data)

# Create a DataFrame from the output data
# output_df = pd.DataFrame(output_data)

# Save to CSV
# output_df.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file}")
