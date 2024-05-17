import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Функция для генерации данных
def generate_data(num_samples, num_features=3):
    features = []
    for _ in range(num_features):
        feature = np.random.normal(loc=0, scale=1, size=num_samples)
        features.append(feature)
    df = pd.DataFrame(features).T
    df.columns = ['feature_{}'.format(i) for i in range(1, num_features + 1)]
    return df


# Функция для предобработки данных
def preprocess_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)


# Функция для разделения данных на обучающую и тестовую выборки
def split_data(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df


# Основной блок кода
if __name__ == "__main__":
    num_samples = 1000
    num_features = 5

    data = generate_data(num_samples, num_features)

    # Предобработка данных
    preprocessed_data = preprocess_data(data)

    # Разделение данных на обучающую и тестовую выборки
    train_data, test_data = split_data(preprocessed_data)

    # Сохранение данных в CSV-файлы
    train_data.to_csv('train/preprocessed_data.csv', index=False)
    test_data.to_csv('test/preprocessed_data.csv')

    print("Данные успешно сгенерированы, предобработаны и сохранены.")
