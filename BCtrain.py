import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import HeNormal


df = pd.read_csv('./data/training_data.csv')

print(df['Facade_Orientation'].unique())
df = pd.get_dummies(df, columns=['Area', 'Insulation'], drop_first=True)

df = pd.get_dummies(df, columns=['Facade_Orientation'], drop_first=False)

filters = {"Facade_Orientation_S": "True"}
print("Df length before filters: ", len(df))
for parameter, value in filters.items():
    if parameter in df.columns:
        print(f"Applying filter: {parameter} == {value}")
        df = df[df[parameter].astype(str) == str(value)]
print("Df length after filters: ", len(df))

df.drop(columns=['Hour'], inplace=True)
df.fillna(df.median(), inplace=True)

scaler = StandardScaler()
X_prescaled = df.drop('Temperature', axis=1)  # Features
print(X_prescaled.info())
X = scaler.fit_transform(X_prescaled)

target_scaler = StandardScaler()
y_prescaled = df['Temperature']  # Target
print(y_prescaled.info())
y = target_scaler.fit_transform(y_prescaled.values.reshape(-1, 1))  # Reshape y to be a 2D array for scaling


# Check for NaN or infinite values
if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    print("NaN values detected in data!")
    
if np.any(np.isinf(X)) or np.any(np.isinf(y)):
    print("Infinite values detected in data!")


# print(df.info())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

callbacks=[
    tf.keras.callbacks.TensorBoard(
        log_dir="./logs",
        histogram_freq=0,  # How often to log histogram visualizations
        embeddings_freq=0,  # How often to log embedding visualizations
        update_freq="epoch",
    ),
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=20,
        verbose=1,
    )
]

# DEFINE MODEL

model = Sequential([
    Dense(512, activation='relu', input_dim=X.shape[1], kernel_initializer=HeNormal()), 
    Dropout(0.2),
    Dense(256, activation='relu', kernel_initializer=HeNormal()),
    Dense(128, activation='relu', kernel_initializer=HeNormal()),
    Dense(64, activation='relu', kernel_initializer=HeNormal()),
    Dense(1)  # Output layer for regression
])

model.summary()

# COMPILE MODEL

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.6,
    decay_steps=10000,
    decay_rate=0.9)

loss_fn = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.legacy.SGD(learning_rate = lr_schedule, momentum = 0.6)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.5e-5)

model.compile(optimizer = optimizer, loss=loss_fn, metrics=["mae"])

# TRAIN

model.fit(X_train, y_train, verbose = 1, batch_size=100, epochs=300, callbacks=callbacks, validation_data=(X_test, y_test))

model.save("BCmodel_South_only.h5")

score = model.evaluate(X_test, y_test, verbose=True)
print("Test MSE:", score[0])
print("Test MAE:", score[1])

hour = 14.0

example_data = {
    'Rain': 0.0,
    'Solar Rad': 300.0,
    'Prędkość wiatru  [m/s]': 2.5,
    'Temperatura powietrza [°C]': 18.0,
    'Wilgotność względna [%]': 65.0,
    'Ciśnienie na poziomie stacji [hPa]': 1000.0,
    'Floor_Level': 1.0,
    'Area_green': True,
    'Insulation_uninsulated': True,
    'Facade_Orientation_E': False,
    'Facade_Orientation_N': True,
    'Facade_Orientation_S': False,
    'Facade_Orientation_W': False,
    'Hour_Cos': np.sin(2 * np.pi * hour / 24),
    'Hour_Sin': np.cos(2 * np.pi * hour / 24)
}



example_df = pd.DataFrame([example_data])
example_df = example_df.astype(float)

example_scaled = scaler.transform(example_df)

predicted_temperature = model.predict(example_scaled)

predicted_temperature_original = target_scaler.inverse_transform(predicted_temperature)

print("Predicted Temperature (Original Scale):", predicted_temperature_original.flatten())