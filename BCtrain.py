import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2


data_path="./data/training_data.csv"
columns_to_scale = ['Rain', 'Prędkość wiatru  [m/s]', 'Temperatura powietrza [°C]', 'Wilgotność względna [%]', 'Ciśnienie na poziomie stacji [hPa]', 'SunDirectionConv']


def scaleData(df):
    df.drop(columns=['Hour'], inplace=True)
    
    # features, ie imputs to the model
    scaler = StandardScaler()
    X_prescaled = df.drop('Temperature', axis=1)
    X_prescaled = X_prescaled.drop('Solar Rad', axis=1)
    X_prescaled[columns_to_scale] = scaler.fit_transform(X_prescaled[columns_to_scale])
    X = X_prescaled

    # target, ie desired output of the model 
    target_scaler = StandardScaler()
    y_prescaled = df['Temperature']
    y = target_scaler.fit_transform(y_prescaled.values.reshape(-1, 1))  # reshape y to be a 2D array for scaling

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, target_scaler


# CALLBACKS ----------------------------------------------------------------
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
    
    # LOAD DATA ------------------------------------------------------------
    df = pd.read_csv('./data/training_data.csv')
    print("Data loaded")
    
    X_train, X_test, y_train, y_test, scaler, target_scaler = scaleData(df)
    
    X_train.drop(columns=['Date'], inplace=True)
    X_test.drop(columns=['Date'], inplace=True)
    
    # DEFINE MODEL ---------------------------------------------------------
    
    # previous model saved for reference
    # model = Sequential([
    #     Dense(512, activation=tf.nn.leaky_relu, input_dim=X_train.shape[1], kernel_initializer=HeNormal()), 
    #     Dense(256, activation=tf.nn.leaky_relu, kernel_initializer=HeNormal()),
    #     Dropout(0.2),
    #     Dense(128, activation=tf.nn.leaky_relu, kernel_initializer=HeNormal()),
    #     Dense(64, activation=tf.nn.leaky_relu, kernel_initializer=HeNormal()),
    #     Dense(1)  # Output layer for regression
    # ])
    
    #final model architecture
    inputs = Input(shape=(X_train.shape[1],))

    x1 = Dense(512, activation=tf.nn.leaky_relu, kernel_initializer=HeNormal())(inputs)
    x2 = Dense(256, activation=tf.nn.leaky_relu, kernel_initializer=HeNormal())(x1)
    x3 = Dense(128, activation=tf.nn.leaky_relu, kernel_initializer=HeNormal())(x2)

    # skip connection layer
    x4 = Dense(128, activation=tf.nn.leaky_relu, kernel_initializer=HeNormal())(inputs)

    x5 = Add()([x3, x4])  # Skip connection: adding output of x4 to x3

    x6 = Dense(64, activation=tf.nn.leaky_relu, kernel_initializer=HeNormal())(x5)
    x7 = Dense(32, activation=tf.nn.leaky_relu, kernel_initializer=HeNormal())(x6)
    x8 = Dense(16, activation=tf.nn.leaky_relu, kernel_initializer=HeNormal())(x7)
    outputs = Dense(1)(x8)

    model = Model(inputs=inputs, outputs=outputs)

    # print model summary
    model.summary()
    
    # COMPILE MODEL ---------------------------------------------------------
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=10000,
        decay_rate=0.9)
    loss_fn = tf.keras.losses.MeanSquaredError()
    # optimizer = tf.keras.optimizers.legacy.SGD(learning_rate = lr_schedule, momentum = 0.6)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    model.compile(optimizer = optimizer, loss=loss_fn, metrics=tf.keras.metrics.R2Score())
    
    tf.keras.utils.plot_model(model,
        to_file="B&Cmodel.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir="TB",
        expand_nested=False,
        dpi=200,
        show_layer_activations=True,
        show_trainable=False,
    )

    # TRAIN MODEL ---------------------------------------------------------
    model.fit(X_train, y_train, verbose = 1, batch_size=100, epochs=100, callbacks=callbacks, validation_data=(X_test, y_test))

    # SAVE MODEL ---------------------------------------------------------
    model.save("BCmodel.h5")