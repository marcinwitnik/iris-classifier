import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

def train_model(model, X_train, y_train):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[tensorboard_callback]
    )

    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Dokładność modelu')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Strata modelu')
    plt.legend()
    plt.grid(True)
    plt.show()

    return history
