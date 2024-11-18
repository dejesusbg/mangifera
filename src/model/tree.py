from sklearn.tree import DecisionTreeClassifier
from . import mango


class MangoTree(mango):
    def __init__(self, X, y, class_weight=None):
        super().__init__(X, y)

        self.model = DecisionTreeClassifier(
            criterion="entropy",
            splitter="best",
            max_depth=1000,
            class_weight=class_weight,
        )

        print("Model initialized successfully.")

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")
