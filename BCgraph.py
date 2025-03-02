import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from BCtrain import scaleData, columns_to_scale

# read dataset, apply filters, recreate the validation split and make predictions 
def filter_and_predict(data_path, model_path, filters=None):

    df = pd.read_csv(data_path)
    
    if filters:
        print("Df length before filters: ", len(df))
        for parameter, value in filters.items():
            if parameter in df.columns:
                print(f"Applying filter: {parameter} == {value}")
                df = df[df[parameter].astype(str) == str(value)]
        print("Df length after filters: ", len(df))
    
    X_train, X_test, y_train, y_test, scaler, target_scaler = scaleData(df)
    
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    X_test_copy.drop(columns=['Date'], inplace=True)
    X_train_copy.drop(columns=['Date'], inplace=True)
    # load the pre-trained model and make predictions
    model = load_model(model_path)
    predicted_output = model.predict(X_test_copy)
    
    # inverse transform predictions and input values to their original scales
    predicted_output_original = target_scaler.inverse_transform(predicted_output)
    true_output_original = target_scaler.inverse_transform(y_test)
    X_test[columns_to_scale] = scaler.inverse_transform(X_test[columns_to_scale])
    X_train[columns_to_scale] = scaler.inverse_transform(X_train[columns_to_scale])
    
    return model, X_test, X_test_copy, predicted_output_original, true_output_original
    
def plot(predicted_output_original, true_output_original, X_test, filters, model_path, plot_type):
    X_test_df = pd.DataFrame(X_test)
    
    hours = np.arctan2(X_test_df['Hour_Sin'], X_test_df['Hour_Cos']) * 24 / (2 * np.pi)
    hours = hours % 24
    meteo_temp = X_test_df['Temperatura powietrza [°C]']
    print("Meteo temp: ", meteo_temp)
    
    plot_df = pd.DataFrame({
        "Hour": round(hours, 0),
        "Predicted_Temperature": predicted_output_original.flatten(),
        "True_Temperature": true_output_original.flatten(),
        "Meteo_Temperature": meteo_temp
    })
    
    # Prepare data for plotting
    plot_data = []
    for hour in sorted(plot_df["Hour"].unique()):  
        hourly_data = plot_df[plot_df["Hour"] == hour]  
        plot_data.append(pd.DataFrame({
            "Hour": [hour] * len(hourly_data),
            "Type": ["True Temperature"] * len(hourly_data),
            "Value": hourly_data["True_Temperature"].values,
        }))
        plot_data.append(pd.DataFrame({
            "Hour": [hour] * len(hourly_data),
            "Type": ["Predicted Temperature"] * len(hourly_data),
            "Value": hourly_data["Predicted_Temperature"].values,
        }))
        plot_data.append(pd.DataFrame({
            "Hour": [hour] * len(hourly_data),
            "Type": ["Meteo Temperature"] * len(hourly_data),
            "Value": hourly_data["Meteo_Temperature"].values,
        }))
    
    # concatenate all hourly data into one DataFrame
    plot_df = pd.concat(plot_data, ignore_index=True)
    
    # generate a filter description for the title
    filter_description = " | ".join([f"{k}: {v}" for k, v in filters.items()]) if isinstance(filters, dict) else "No filters applied"
            
    if plot_type == "box":
        plt.figure(figsize=(16, 10))
        sns.boxplot(
            data=plot_df,
            x="Hour",
            y="Value",
            hue="Type",
            palette={"True Temperature": "#7AB", "Predicted Temperature": "orange", "Meteo Temperature": "green"}
        )
        
        plt.title(f"Box Plot of Temperatures by Hour of the Day\nFilters: {filter_description}\nModel: {model_path}, Sample size: {int(np.floor(len(plot_df)/3))}", fontsize=16)
        plt.xlabel("Hour of the Day", fontsize=14)
        plt.ylabel("Temperature (°C)", fontsize=14)
        plt.legend(title="Temperature Type", fontsize=12, title_fontsize=14)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
            
    if plot_type == "line":
        plt.figure(figsize=(16, 10))
        sns.lineplot(
            data=plot_df,
            x="Hour",
            y="Value",
            hue="Type",
            palette={"True Temperature": "#7AB", "Predicted Temperature": "orange", "Meteo Temperature": "green"}
        )
        
        # Customize plot
        plt.title(f"Plot of Temperatures by Hour of the Day\nFilters: {filter_description}\nModel: {model_path}, Sample size: {int(np.floor(len(plot_df)/3))}", fontsize=16)
        plt.xlabel("Hour of the Day", fontsize=14)
        plt.ylabel("Temperature (°C)", fontsize=14)
        plt.legend(title="Temperature Type", fontsize=12, title_fontsize=14)
        plt.grid(alpha=0.3)
        
        # Show the plot
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    data_path="./data/training_data.csv"
    model_path="BCmodel.h5"
    filters = None
    # filters = {"Facade_Orientation_N": 1}
    # filters = {"Facade_Orientation_S": 1}
    # filters = {"Facade_Orientation_E": 1}
    # filters = {"Facade_Orientation_W": 1}
    # filters = {"Date": "2024-08-12"}
    # filters = {"Floor_Level": 0}
    # filters = {"Date": "2024-08-13", "Facade_Orientation_N": 1}
    # filters = {"Date": "2024-08-13", "Facade_Orientation_S": 1}
    # filters = {"Date": "2024-08-13", "Facade_Orientation_E": 1}
    # filters = {"Date": "2024-08-13", "Facade_Orientation_W": 1}
    # filters = {"Date": "2024-07-11", "Area_green": 1}
    filters = {"Date": "2024-07-11", "Area_concrete": 1}
    
    model, X_test, X_test_scaled, predicted_output_original, true_output_original = filter_and_predict(data_path, model_path, filters)
    plot(predicted_output_original, true_output_original, X_test, filters, model_path, plot_type="line")