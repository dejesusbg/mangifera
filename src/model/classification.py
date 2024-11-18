import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class MangoClassificationModel:
    MODEL_DIR = "../assets/models"
    IMAGE_DIR = "../assets/images"

    def __init__(self, X, y):
        self.X = X["train"]
        self.y = y["train"]
        self.X_val = X["validation"]
        self.y_val = y["validation"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def run(self):
        """Run the model."""
        self.train()
        self._save_model()

    def predict(self, split):
        """Predict the output of the model on the given split."""
        if split == "test":
            X_true, y_true = self.X_test, self.y_test
        elif split == "validation":
            X_true, y_true = self.X_val, self.y_val

        y_pred = self.model.predict(X_true)
        y_pred = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_true, y_pred) * 100
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        return self._show_metrics(split, accuracy, cm, report)

    def _save_model(self):
        """Save the model to a file."""
        name = type(self).__name__
        file_path = os.path.join(self.MODEL_DIR, f"{name}.joblib")

        if os.path.exists(file_path):
            os.remove(file_path)
        joblib.dump(self.model, file_path)

    @staticmethod
    def _show_metrics(split, accuracy, cm, report):
        """Display the metrics of the model on the given split."""
        cm_index = [f"{i}" for i in range(cm.shape[0])]
        cm_columns = [f"{i}" for i in range(cm.shape[1])]

        cm_df = pd.DataFrame(cm, index=cm_index, columns=cm_columns)
        report_df = pd.DataFrame(report).transpose()

        combined_df = pd.concat([report_df, cm_df], axis=1)
        print(f"- Accuracy ({split}): {accuracy:.2f}%")
        return combined_df.fillna("")

    @classmethod
    def custom_predict(cls, name, image, preprocessor):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)

        model_path = os.path.join(cls.MODEL_DIR, f"{name}.joblib")
        model = joblib.load(model_path)

        image_path = os.path.join(cls.IMAGE_DIR, image)
        image = preprocessor.process_image(image_path)
        image_df = pd.DataFrame(image)
        prediction = model.predict(image_df)
        return prediction
