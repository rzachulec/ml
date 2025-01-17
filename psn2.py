import tensorflow as tf
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# DOWNLOAD TEST DATA

cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# DEFINE CALLBACKS

class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                tf.summary.histogram(f'{layer.name}_weights', layer.kernel, step=epoch)
            # Log the biases of each layer
            # if hasattr(layer, 'bias'):
            #     tf.summary.histogram(f'{layer.name}_biases', layer.bias, step=epoch)

            # Optionally, log the layer errors (e.g., residuals)
            # For this, you would need to compute the error (layer output - expected output)
            # You can use the following as an example of logging layer output (error as the residual)
            # if hasattr(layer, 'output'):
            #     residuals = layer.output - self.model.output
            #     tf.summary.histogram(f'{layer.name}_errors', residuals, step=epoch)


callbacks=[
    CustomTensorBoard(
        log_dir="./logs",
        histogram_freq=10,  # Log histograms every epoch
        embeddings_freq=0,  # Skip embedding visualizations
        update_freq=10,
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

model = tf.keras.applications.ResNet50V2(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,
)

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
    initial_learning_rate=0.03,
    decay_steps=10000,
    decay_rate=0.9)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)

model.compile(optimizer = optimizer, loss=loss_fn, metrics=["acc"])

# TRAIN

model.fit(x_train, y_train, verbose = 1, batch_size=1000, epochs=10, callbacks=callbacks, validation_split=0.2)

# score = model.evaluate(X_test, y_test_encoded, verbose=True)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])