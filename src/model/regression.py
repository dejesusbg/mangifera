from sklearn.linear_model import Ridge
from . import mango


class MangoRegression(mango):
    def __init__(self, X, y, param_grid=None):
        super().__init__(X, y)

        self.model = Ridge()
        self.param_grid = param_grid
        print("Model initialized successfully.")

    def train(self):
        self.grid_search()
