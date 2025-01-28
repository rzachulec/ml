import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def filter_and_predict(data_path, model_path, filters=None):
    # Load the dataset
    df = pd.read_csv(data_path)

    df = pd.get_dummies(df, columns=['Facade_Orientation'], drop_first=False)
    df.fillna(df.median(), inplace=True)
    
    if filters:
        print("Df length before filters: ", len(df))
        for parameter, value in filters.items():
            if parameter in df.columns:
                print(f"Applying filter: {parameter} == {value}")
                df = df[df[parameter].astype(str) == str(value)]
        print("Df length after filters: ", len(df))
    
    # Extract the 'Hour' column for plotting and drop it from the main DataFrame
    hour_df = df['Hour']
    df.drop(columns=['Hour'], inplace=True)
    
    print("Shape before one-hot encoding:", df.shape)
    
    # Preprocess categorical features
    # df = pd.get_dummies(df, columns=['Area', 'Insulation'], drop_first=True)
    
    print("Shape after one-hot encoding:", df.shape)   
    
    # Scale the features and target
    scaler = StandardScaler()
    X_prescaled = df.drop('Temperature', axis=1)  # Features for scaling
    X = scaler.fit_transform(X_prescaled)
    
    target_scaler = StandardScaler()
    y_prescaled = df['Temperature']
    y = target_scaler.fit_transform(y_prescaled.values.reshape(-1, 1))  # Reshape y to 2D for scaling
    
    # Load the pre-trained model and make predictions
    model = load_model(model_path)
    predicted_output = model.predict(X)
    
    # Inverse transform predictions and true target values to their original scales
    predicted_output_original = target_scaler.inverse_transform(predicted_output)
    true_output_original = target_scaler.inverse_transform(y)
    
    # Add predictions, true values, and hours back to the DataFrame for plotting
    df["Hour"] = hour_df.reset_index(drop=True)  # Re-add hour column
    df["True_Temperature"] = true_output_original.flatten()
    df["Predicted_Temperature"] = predicted_output_original.flatten()
    df["Meteo_Temperature"] = df["Temperatura powietrza [°C]"]  # Rename feature temperature to Meteo Temperature
    
    # Prepare data for plotting
    box_plot_data = []
    for hour in sorted(hour_df.unique()):  # Use `hour_df` for the unique hour values
        hourly_data = df[df["Hour"] == hour]  # Filter rows by the current hour
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
    
    # Generate a filter description for the title
    filter_description = " | ".join([f"{k}: {v}" for k, v in filters.items()]) if filters else "No filters applied"
    
    # Plot the box plot using seaborn
    plt.figure(figsize=(16, 10))
    sns.boxplot(
        data=box_plot_df,
        x="Hour",
        y="Value",
        hue="Type",
        palette={"True Temperature": "#7AB", "Predicted Temperature": "orange", "Meteo Temperature": "green"}
    )
    
    # Customize plot
    plt.title(f"Box Plot of Temperatures by Hour of the Day\nFilters: {filter_description}\n Model: {model_path}", fontsize=16)
    plt.xlabel("Hour of the Day", fontsize=14)
    plt.ylabel("Temperature (°C)", fontsize=14)
    plt.legend(title="Temperature Type", fontsize=12, title_fontsize=14)
    plt.grid(alpha=0.3)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

filters = {"Facade_Orientation_S": "True"}
filter_and_predict(
    data_path="./data/training_data.csv",
    model_path="BCmodel_South_only.h5",
    filters=filters
)