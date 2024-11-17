from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

class MangoTree:
    def __init__(self,x ,y):
        self.X = x["train"]
        self.y = y["train"]
        self.X_val = x["validation"]
        self.y_val = y["validation"]

        self.model= DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=1000,class_weight='balanced')
        print("Model initialized successfully.")
        self.model.fit(self.X, self.y)
        print("Model trained successfully.")
        
    # Evaluar el modelo en el conjunto de prueba
    def test(self):
        y_pred = self.model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(self.y_val, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(self.y_val, y_pred))
