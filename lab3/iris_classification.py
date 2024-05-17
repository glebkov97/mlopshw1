# iris_classification.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd


def prepare_data():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    predictions = model.predict(X_test)
    return predictions


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_model(X_train, y_train)
    predictions = predict(model, X_test)
    print(predictions)
