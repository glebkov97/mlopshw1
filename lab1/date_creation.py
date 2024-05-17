import numpy as np
import pandas as pd


def generate_data(num_samples, num_features=3):
    """
    Генерирует случайные данные для заданного количества образцов и признаков.

    :param num_samples: Количество образцов данных.
    :param num_features: Количество признаков в каждом образце.
    :return: DataFrame с сгенерированными данными.
    """
    # Генерация случайных значений для каждого признака
    features = []
    for _ in range(num_features):
        feature = np.random.normal(loc=0, scale=1, size=num_samples)
        features.append(feature)

    # Создание DataFrame из сгенерированных данных
    df = pd.DataFrame(features).T
    df.columns = ['feature_{}'.format(i) for i in range(1, num_features + 1)]

    return df


def split_data(df, train_ratio=0.8):
    """
    Разделяет DataFrame на обучающую и тестовую выборки согласно указанному соотношению.

    :param df: Исходный DataFrame с данными.
    :param train_ratio: Соотношение размеров обучающей и тестовой выборок.
    :return: Обучающая и тестовая выборки.
    """
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    return train_df, test_df


if __name__ == "__main__":
    # Указываем количество образцов и признаков
    num_samples = 1000
    num_features = 5

    # Генерируем данные
    data = generate_data(num_samples, num_features)

    # Разделяем данные на обучающую и тестовую выборки
    train_data, test_data = split_data(data)

    # Сохраняем данные в соответствующие папки
    train_data.to_csv('train/data.csv', index=False)
    test_data.to_csv('test/data.csv', index=False)

    print("Данные успешно сгенерированы и сохранены.")

