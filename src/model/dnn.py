import matplotlib.pyplot as plt
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from . import mango


class MangoDeepNetwork(mango):
    def __init__(self, X, y, epochs=20, batch_size=32, class_weight=None):
        super().__init__(X, y)

        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight

        layers = [
            Dense(128, input_dim=self.X_train.shape[1], activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]

        self.model = Sequential(layers)

        self.model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        print("Model initialized successfully.")

    def train(self):
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=self.class_weight,
            validation_data=(self.X_val, self.y_val),
            verbose=False,
        )

        print("Model trained successfully.")
        self.plot_history(history)

    def plot_history(self, history):
        """Plot the history of the loss and accuracy over epochs."""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training")
        plt.plot(history.history["val_loss"], label="Validation")
        plt.title("Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (pérdida)")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Training")
        plt.plot(history.history["val_accuracy"], label="Validation")
        plt.title("Accuracy over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()
