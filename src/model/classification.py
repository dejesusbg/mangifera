import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class MangoClassificationModel:
    MODEL_DIR = "../assets/models"
    IMAGE_DIR = "../assets/images"

    def __init__(self, X, y):
        self.X_train = X["train"]
        self.y_train = y["train"]
        self.X_val = X["validation"]
        self.y_val = y["validation"]

    def run(self):
        """Run the model."""
        self.train()
        self._save_model()

    def grid_search(self):
        """Perform grid search and train the model."""
        param_grid = self.param_grid

        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(self.X_train, self.y_train)

        self.model = grid_search.best_estimator_
        best_param = grid_search.best_params_

        print(f"Model trained successfully with params {best_param}.")

    def predict(self, split):
        """Predict the output of the model on the given split."""
        if split == "train":
            X_true, y_true = self.X_train, self.y_train
        if split == "validation":
            X_true, y_true = self.X_val, self.y_val

        try:
            y_pred = self.model.predict(X_true, verbose=0)
        except TypeError:
            y_pred = self.model.predict(X_true)

        y_pred = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": accuracy,
            "precision-0": report["0"]["precision"],
            "precision-1": report["1"]["precision"],
            "recall-0": report["0"]["recall"],
            "recall-1": report["1"]["recall"],
            "f1-0": report["0"]["f1-score"],
            "f1-1": report["1"]["f1-score"],
            "cm": cm,
        }

        return self._show_metrics(split, accuracy, cm, report), metrics

    def _save_model(self):
        """Save the model to a file."""
        name = type(self).__name__
        file_path = os.path.join(self.MODEL_DIR, f"{name}.joblib")
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        if os.path.exists(file_path):
            os.remove(file_path)
        joblib.dump(self.model, file_path)

    @staticmethod
    def _show_metrics(split, accuracy, cm, report):
        """Display the metrics of the model on the given split."""
        print(f"- Accuracy ({split}): {accuracy:.4f}")

        cm_index = [f"{i}" for i in range(cm.shape[0])]
        cm_columns = [f"{i}" for i in range(cm.shape[1])]

        cm_df = pd.DataFrame(cm, index=cm_index, columns=cm_columns)
        report_df = pd.DataFrame(report).transpose()

        combined_df = pd.concat([report_df, cm_df], axis=1)
        return combined_df.fillna("")

    @classmethod
    def custom_predict(cls, name, image, preprocessor):
        """Predict the output of the model on the given image."""
        model_path = os.path.join(cls.MODEL_DIR, f"{name}.joblib")
        model = joblib.load(model_path)

        image_path = os.path.join(cls.IMAGE_DIR, image)
        image = preprocessor.process_image(image_path)
        image_df = pd.DataFrame(image)

        try:
            prediction = model.predict(image_df, verbose=0)
        except TypeError:
            prediction = model.predict(image_df)

        return prediction
