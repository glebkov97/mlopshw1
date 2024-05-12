from sklearn.linear_model import LinearRegression
import numpy as np

def prepare_and_train_model(train_data_path, test_data_path):
    X_train = np.arange(len(np.loadtxt(train_data_path, delimiter=","))).reshape(-1, 1)
    y_train = np.loadtxt(train_data_path, delimiter=",")

    X_test = np.arange(len(np.loadtxt(test_data_path, delimiter=","))).reshape(-1, 1)
    y_test = np.loadtxt(test_data_path, delimiter=",")

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Оценка модели
    score = model.score(X_test, y_test)
    print(f"Model accuracy: {score}")


if __name__ == "__main__":
    train_data_path = "train_scaled.csv"
    test_data_path = "test_scaled.csv"

    prepare_and_train_model(train_data_path, test_data_path)
