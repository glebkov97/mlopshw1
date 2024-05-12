from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data


if __name__ == "__main__":
    train_data_path = "train/data_set_0.csv"
    test_data_path = "test/data_set_0.csv"

    train_scaled = preprocess_data(train_data_path)
    test_scaled = preprocess_data(test_data_path)

    # Сохраняем обработанные данные
    np.savetxt("train_scaled.csv", train_scaled, delimiter=",")
    np.savetxt("test_scaled.csv", test_scaled, delimiter=",")
