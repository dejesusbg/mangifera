from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class MangoRegression:
    def __init__(self, x, y):

        self.X = x["train"]
        self.y = y["train"]
        self.X_val = x["validation"]
        self.y_val = y["validation"]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Inicializar el modelo de regresión Ridge
        self.model = Ridge()
        print("Model initialized successfully.")

    def run(self):
        self.train()
        self.test()
        self.validate()

    def train(self):
        # Aplicar GridSearchCV para encontrar el mejor valor de alpha
        param_grid = {'alpha': [1.65, 1.66, 1.67, 1.675, 1.68, 1.685, 1.69, 1.695, 1.7, 1.705, 1.71, 1.715, 1.72, 1.725, 1.73]}
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2')
        grid_search.fit(self.X_train, self.y_train)

        # Asignar el mejor modelo encontrado
        self.model = grid_search.best_estimator_
        print(f"Model trained successfully with best alpha: {grid_search.best_params_['alpha']}")

    def test(self):
        y_pred = self.model.predict(self.X_test)
        score_r2 = r2_score(self.y_test, y_pred)
        score_RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))

        print(f"R^2 Score: {score_r2:.4f}")
        print(f"Root Mean Squared Error (RMSE): {score_RMSE:.4f}")

    def validate(self):
        # Validar el modelo con el conjunto de validación externo
        y_pred_val = self.model.predict(self.X_val)

        # Evaluar el rendimiento en el conjunto de validación
        score_r2_val = r2_score(self.y_val, y_pred_val)
        score_rmse_val = np.sqrt(mean_squared_error(self.y_val, y_pred_val))

        print("\nValidation Results:")
        print(f"R^2 Score: {score_r2_val:.4f}")
        print(f"Root Mean Squared Error (RMSE): {score_rmse_val:.4f}")
