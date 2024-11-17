from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from . import Mango


class MangoNetwork(Mango):
    def __init__(self, X, y, max_iter=20, batch_size=32):
        super().__init__(X, y)

        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 32),
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            batch_size=batch_size,
            random_state=42,
            verbose=False,
        )

        print("Model initialized successfully.")

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def validation(self):
        y_pred = self.model.predict(self.X_val)
        return self.get_metrics(y_pred)
