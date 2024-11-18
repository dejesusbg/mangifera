from sklearn.neural_network import MLPClassifier
from . import mango


class MangoNetwork(mango):
    def __init__(self, X, y, param_grid=None):
        super().__init__(X, y)

        self.model = MLPClassifier(activation="relu", solver="adam", random_state=42)
        self.param_grid = param_grid
        print("Model initialized successfully.")

    def train(self):
        self.grid_search()
