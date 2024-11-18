from sklearn.ensemble import RandomForestClassifier
from . import mango


class MangoForest(mango):
    def __init__(self, X, y):
        super().__init__(X, y)

        self.model = RandomForestClassifier(random_state=42)
        print("Model initialized successfully.")

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")
