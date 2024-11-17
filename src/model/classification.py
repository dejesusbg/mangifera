import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tf_keras.models import load_model


class MangoClassificationModel:
    MODEL_DIR = "../assets/"

    def __init__(self, X, y):
        self.X = X["train"]
        self.y = y["train"]
        self.X_val = X["validation"]
        self.y_val = y["validation"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def run(self):
        self.train()
        self.save_model()
        return self.validation()

    def get_metrics(self, y_pred):
        y_pred = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(self.y_val, y_pred)
        report = classification_report(self.y_val, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_val, y_pred)
        return self.show_metrics(accuracy, cm, report)

    def show_metrics(self, accuracy, cm, report):
        cm_index = [f"{i}" for i in range(cm.shape[0])]
        cm_columns = [f"{i}" for i in range(cm.shape[1])]

        cm_df = pd.DataFrame(cm, index=cm_index, columns=cm_columns)
        report_df = pd.DataFrame(report).transpose()
        combined_df = pd.concat([report_df, cm_df], axis=1)

        print(f"Accuracy: {accuracy:.4f}")
        return combined_df.fillna("")

    def save_model(self):
        name = type(self).__name__

        if name == "MangoDNN":
            self.model.save(f"../assets/{name}.h5")
        else:
            joblib.dump(self.model, f"../assets/{name}.joblib")

    @classmethod
    def custom_predict(cls, name, image, preprocessor):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)

        if name == "MangoDNN":
            path = os.path.join(cls.MODEL_DIR, f"{name}.h5")
            model = load_model(path)
        else:
            path = os.path.join(cls.MODEL_DIR, f"{name}.joblib")
            model = joblib.load(path)

        image_path = os.path.join(cls.MODEL_DIR, image)
        input_data = preprocessor.process_image(image_path)
        input_df = pd.DataFrame(input_data)
        prediction = model.predict(input_df)
        label = "Ripe" if prediction < 0.5 else "Rotten"
        return label
