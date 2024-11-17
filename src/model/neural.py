from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tf_keras.models import Sequential
from tf_keras.layers import Dense


class MangoNetwork:
    def __init__(self, X, y, epochs=20, batch_size=32, class_weight=None):
        self.X = X["train"]
        self.y = y["train"]
        self.X_val = X["validation"]
        self.y_val = y["validation"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.0942, random_state=42
        )

        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.model = self._build_model(self.X_train.shape[1])
        print("Model initialized successfully.")

    def train(self):
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=self.class_weight,
            verbose=1,
        )

        print("Model trained successfully.")

    def test(self):
        y_pred_test = self.model.predict(self.X_test)
        y_pred_test = (y_pred_test > 0.5).astype(int)
        self._metrics(self.y_test, y_pred_test)

    def validate(self):
        y_pred_val = self.model.predict(self.X_val)
        y_pred_val = (y_pred_val > 0.5).astype(int)
        self._metrics(self.y_val, y_pred_val)

    @staticmethod
    def _build_model(input_dim):
        model = Sequential(
            [
                Dense(128, input_dim=input_dim, activation="relu"),
                Dense(32, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        return model

    @staticmethod
    def _metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
