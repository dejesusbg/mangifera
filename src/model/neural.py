from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tf_keras.models import Sequential
from tf_keras.layers import Dense


class MangoNetwork:
    def __init__(self, x, y):
        from sklearn.preprocessing import LabelEncoder

        # Suponiendo que tus etiquetas están en la variable `y`
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        # Initialize the neural network model
        model = Sequential()

        # Suponiendo que tus etiquetas están en la variable `y`
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        # Add layers: input layer, one or more hidden layers, and output layer
        model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        # Compile the model
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy", "mse", "mae"],
        )

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        print("Model trained successfully.")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(
            int
        )  # Convert probabilities to binary class (0 or 1)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
