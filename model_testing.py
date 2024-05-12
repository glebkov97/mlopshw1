from sklearn.metrics import mean_squared_error
import numpy as np


def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    model_path = "model.pkl"
    X_test = np.arange(len(np.loadtxt("test_scaled.csv", delimiter=","))).reshape(-1, 1)
    y_test = np.loadtxt("test_scaled.csv", delimiter=",")

    # Загрузка обученной модели
    model = LinearRegression()
    model.load(model_path)

    test_model(model, X_test, y_test)
