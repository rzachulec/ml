import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
# from tensorflow.keras import 


data_path="./data/training_data.csv"
columns_to_scale = ['Rain', 'Prędkość wiatru  [m/s]', 'Temperatura powietrza [°C]', 'Wilgotność względna [%]', 'Ciśnienie na poziomie stacji [hPa]', 'SunDirectionConv']


def scaleData(df):
    df.drop(columns=['Hour'], inplace=True)
    
    # Features
    scaler = StandardScaler()
    X_prescaled = df.drop('Temperature', axis=1)
    X_prescaled = X_prescaled.drop('Solar Rad', axis=1)
    X_prescaled[columns_to_scale] = scaler.fit_transform(X_prescaled[columns_to_scale])
    X = X_prescaled

    # Target     
    target_scaler = StandardScaler()
    y_prescaled = df['Temperature']
    y = target_scaler.fit_transform(y_prescaled.values.reshape(-1, 1))  # Reshape y to be a 2D array for scaling


    # # Check for NaN or infinite values
    # if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    #     print("NaN values detected in data!")
        
    # if np.any(np.isinf(X)) or np.any(np.isinf(y)):
    #     print("Infinite values detected in data!")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, target_scaler


# CALLBACKS
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
        patience=30,
        verbose=1,
    )
]


if __name__ == "__main__":
    
    # LOAD DATA
    df = pd.read_csv('./data/training_data.csv')
    print("Data loaded")
    
    X_train, X_test, y_train, y_test, scaler, target_scaler = scaleData(df)
    
    X_train.drop(columns=['Date'], inplace=True)
    X_test.drop(columns=['Date'], inplace=True)
    
    # DEFINE MODEL
    # model = Sequential([
    #     Dense(512, activation='leaky_relu', input_dim=X_train.shape[1], kernel_initializer=HeNormal()), 
    #     Dense(256, activation='leaky_relu', kernel_initializer=HeNormal()),
    #     Dropout(0.2),
    #     Dense(128, activation='leaky_relu', kernel_initializer=HeNormal()),
    #     Dense(64, activation='leaky_relu', kernel_initializer=HeNormal()),
    #     Dense(1)  # Output layer for regression
    # ])
    
        # Input layer
    inputs = Input(shape=(X_train.shape[1],))

    # First layer with Leaky ReLU
    x1 = Dense(512, activation='leaky_relu', kernel_initializer=HeNormal())(inputs)
    
    # Second layer with Leaky ReLU
    x6 = Dense(256, activation='leaky_relu', kernel_initializer=HeNormal())(x1)

    # Second layer with Leaky ReLU
    x2 = Dense(256, activation='leaky_relu', kernel_initializer=HeNormal())(inputs)

    # Add skip connection from the first layer to the third layer
    x3 = Add()([x2, x6])  # Skip connection: adding output of x1 to x2

    # Third layer with Leaky ReLU
    x4 = Dense(128, activation='leaky_relu', kernel_initializer=HeNormal())(x3)

    # Fourth layer with Leaky ReLU
    x5 = Dense(64, activation='leaky_relu', kernel_initializer=HeNormal())(x4)

    # Output layer for regression
    outputs = Dense(1)(x5)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    model.summary()
    
    # COMPILE MODEL
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=10000,
        decay_rate=0.9)
    loss_fn = tf.keras.losses.MeanSquaredError()
    # optimizer = tf.keras.optimizers.legacy.SGD(learning_rate = lr_schedule, momentum = 0.6)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    model.compile(optimizer = optimizer, loss=loss_fn, metrics=tf.keras.metrics.R2Score())

    # TRAIN MODEL
    model.fit(X_train, y_train, verbose = 1, batch_size=100, epochs=100, callbacks=callbacks, validation_data=(X_test, y_test))

    # SAVE MODEL
    model.save("BCmodel.h5")

    # VERIFY
    score = model.evaluate(X_test, y_test, verbose=True)
    print("Test MSE:", score[0])
    print("Test R2:", score[1])