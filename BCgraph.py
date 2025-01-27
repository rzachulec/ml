import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import seaborn as sns



# Assuming your CSV file is named 'data.csv'
df = pd.read_csv('building_data.csv')

# print(df['Facade_Orientation'].unique())
df = pd.get_dummies(df, columns=['Area', 'Insulation'], drop_first=True)

df = pd.get_dummies(df, columns=['Facade_Orientation'], drop_first=False)

df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
df['Time'].fillna(pd.to_datetime('00:00:00'), inplace=True)
df['Hour'] = df['Time'].dt.hour  # Extract hour from the time
df.drop(columns=['Time'], inplace=True)

numerical_cols = ['Rain', 'Solar Rad', 'Prędkość wiatru  [m/s]', 'Temperatura powietrza [°C]', 
                  'Wilgotność względna [%]', 'Ciśnienie na poziomie stacji [hPa]', 'Floor_Level', 'Hour']


df.fillna(df.median(), inplace=True)

scaler = StandardScaler()
X_prescaled = df.drop('Temperature', axis=1)  # Features
# print(X_prescaled.info())
X = scaler.fit_transform(X_prescaled)

target_scaler = StandardScaler()
y_prescaled = df['Temperature']  # Target
# print(y_prescaled.info())
y = target_scaler.fit_transform(y_prescaled.values.reshape(-1, 1))  # Reshape y to be a 2D array for scaling

# Load your saved model
model = load_model("BCmodel.h5")

# Use the model to make predictions on the training data
predicted_output = model.predict(X)

# Inverse transform predictions and true target values to their original scales
predicted_output_original = target_scaler.inverse_transform(predicted_output)
true_output_original = target_scaler.inverse_transform(y)

# Add predictions and true values back to the original DataFrame for grouping
df["True_Temperature"] = true_output_original.flatten()
df["Predicted_Temperature"] = predicted_output_original.flatten()
df["Meteo_Temperature"] = df["Temperatura powietrza [°C]"]  # Rename feature temperature to Meteo Temperature


# Prepare data for plotting
box_plot_data = []

for hour in sorted(df["Hour"].unique()):  # Iterate through each hour
    hourly_data = df[df["Hour"] == hour]
    box_plot_data.append(pd.DataFrame({
        "Hour": [hour] * len(hourly_data),
        "Type": ["True Temperature"] * len(hourly_data),
        "Value": hourly_data["True_Temperature"].values,
    }))
    box_plot_data.append(pd.DataFrame({
        "Hour": [hour] * len(hourly_data),
        "Type": ["Predicted Temperature"] * len(hourly_data),
        "Value": hourly_data["Predicted_Temperature"].values,
    }))
    box_plot_data.append(pd.DataFrame({
        "Hour": [hour] * len(hourly_data),
        "Type": ["Meteo Temperature"] * len(hourly_data),
        "Value": hourly_data["Meteo_Temperature"].values,
    }))

# Concatenate all hourly data into one DataFrame
box_plot_df = pd.concat(box_plot_data, ignore_index=True)

# Plot the box plot using seaborn
plt.figure(figsize=(16, 10))
sns.boxplot(
    data=box_plot_df,
    x="Hour",
    y="Value",
    hue="Type",
    palette={"True Temperature": "blue", "Predicted Temperature": "orange", "Meteo Temperature": "green"}
)

# Customize plot
plt.title("Box Plot of Temperatures by Hour of the Day", fontsize=16)
plt.xlabel("Hour of the Day", fontsize=14)
plt.ylabel("Temperature (°C)", fontsize=14)
plt.legend(title="Temperature Type", fontsize=12, title_fontsize=14)
plt.grid(alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()
