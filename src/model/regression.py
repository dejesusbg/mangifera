from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from . import mango


class MangoRegression(mango):
    def __init__(self, X, y):
        super().__init__(X, y)

        self.model = Ridge()
        print("Model initialized successfully.")

    def train(self):
        param_grid = {
            "alpha": [
                1.65,
                1.66,
                1.67,
                1.675,
                1.68,
                1.685,
                1.69,
                1.695,
                1.7,
                1.705,
                1.71,
                1.715,
                1.72,
                1.725,
                1.73,
            ]
        }

        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(self.X_train, self.y_train)

        self.model = grid_search.best_estimator_
        best_alpha = grid_search.best_params_["alpha"]

        print(f"Model trained successfully with alpha {best_alpha}.")
