from sklearn.tree import DecisionTreeClassifier
from . import mango


class MangoTree(mango):
    def __init__(self, X, y, param_grid=None, class_weight=None):
        super().__init__(X, y)

        self.model = DecisionTreeClassifier(class_weight=class_weight, random_state=42)
        self.param_grid = param_grid
        print("Model initialized successfully.")

    def train(self):
        self.grid_search()
