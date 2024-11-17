from sklearn.tree import DecisionTreeClassifier
from . import Mango


class MangoTree(Mango):
    def __init__(self, X, y):
        super().__init__(X, y)

        self.model = DecisionTreeClassifier(
            criterion="entropy",
            splitter="best",
            max_depth=1000,
            class_weight="balanced",
        )

        print("Model initialized successfully.")

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def validation(self):
        y_pred = self.model.predict(self.X_val)
        return self.get_metrics(y_pred)
