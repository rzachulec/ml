import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import load_model


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


# # Check for NaN or infinite values
# if np.any(np.isnan(X)) or np.any(np.isnan(y)):
#     print("NaN values detected in data!")
    
# if np.any(np.isinf(X)) or np.any(np.isinf(y)):
#     print("Infinite values detected in data!")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = load_model('BCmodel.h5')

# score = model.evaluate(X_test, y_test, verbose=True)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])

example_data = {
    'Rain': 0.0,
    'Solar Rad': 69.0,
    'Prędkość wiatru  [m/s]': 3.0,
    'Temperatura powietrza [°C]': 13.6,
    'Wilgotność względna [%]': 91.0,
    'Ciśnienie na poziomie stacji [hPa]': 995.0,
    'Floor_Level': 1.0,
    'Area_green': True,
    'Insulation_uninsulated': True,
    'Facade_Orientation_E': False,
    'Facade_Orientation_N': False,
    'Facade_Orientation_S': False,
    'Facade_Orientation_W': True,
    'Hour': 8.0
}

example_df = pd.DataFrame([example_data])
example_df = example_df.astype(float)

example_scaled = scaler.transform(example_df)

predicted_temperature = model.predict(example_scaled)

predicted_temperature_original = target_scaler.inverse_transform(predicted_temperature)

print("Original Temperature:", example_data['Temperatura powietrza [°C]'])
print("Predicted Temperature (Original Scale):", predicted_temperature_original.flatten())