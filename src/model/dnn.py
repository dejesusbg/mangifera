from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tf_keras.models import Sequential
from tf_keras.layers import Dense
import tensorflow as tf
from . import Mango


class DeepMangoNetwork(Mango):
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
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=self.class_weight,
            verbose=False,
        )

        print("Model trained successfully.")

    def validation(self):
        y_pred = self.model.predict(self.X_val)
        return self.get_metrics(y_pred)
