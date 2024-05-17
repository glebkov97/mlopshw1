import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Функция для загрузки данных
def load_data(file_path):
    return pd.read_csv(file_path)

# Функция для подготовки данных
def prepare_data(df):
    # Здесь можно добавить код для подготовки данных, например, нормализацию или выбросы
    return df

# Функция для оценки модели
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

# Загрузка данных
data_path = 'test/preprocessed_data.csv'
df = load_data(data_path)

# Подготовка данных
prepared_data = prepare_data(df)

# Предполагаем, что у нас есть функция для получения целевой переменной
y = prepared_data['feature_1']  # Замените 'target' на имя вашей целевой колонки

# Разделение данных на признаки и целевую переменную
X = prepared_data.drop('feature_1', axis=1)

# Загрузка предварительно обученной модели
# model =...  # Здесь должна быть ваша модель, которую вы хотите использовать

# Предсказание на тестовых данных
# mse, r2 = evaluate_model(model, X_test, y_test)  # Предполагается, что X_test и y_test уже определены

print("Модель успешно протестирована.")
