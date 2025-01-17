from pickletools import optimize

import tensorflow as tf
import kagglehub
import pandas as pd
import sklearn

# DOWNLOAD TEST DATA

path = kagglehub.dataset_download("martininf1n1ty/exclusive-xor-dataset")

print("Path to dataset files:", path)

# FORMAT DATA

df = pd.read_csv(path + '/xor.csv')
df.head()

X = df[['X1', 'X2']].to_numpy()
y = df['label'].to_numpy()

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

encoder = sklearn.preprocessing.OneHotEncoder(categories='auto')
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.reshape(-1, 1)).toarray()

# DEFINE CALLBACKS

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

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(4,
                          activation='tanh',
                          kernel_initializer = tf.keras.initializers.VarianceScaling()),
    # tf.keras.layers.Dense(4,
    #                       activation='tanh',
    #                       kernel_initializer = tf.keras.initializers.VarianceScaling()),
    tf.keras.layers.Dense(2,
                          activation='sigmoid',
                          kernel_initializer = tf.keras.initializers.VarianceScaling()),
])

model.summary()

# RENDER MODEL SCHEME

tf.keras.utils.plot_model(
    model,
    to_file="psn1_model.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_layer_activations=True,
    show_trainable=False,
)

# COMPILE MODEL

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.6,
    decay_steps=100,
    decay_rate=0.9)

loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum = 0.6)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)

model.compile(optimizer = optimizer, loss=loss_fn, metrics=["acc"])

# TRAIN

model.fit(X_train, y_train_encoded, verbose = 1, batch_size=100, epochs=1000, callbacks=callbacks, validation_split=0.2)

score = model.evaluate(X_test, y_test_encoded, verbose=True)
print("Test loss:", score[0])
print("Test accuracy:", score[1])