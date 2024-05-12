import numpy as np
import os

def create_data_sets(num_sets=5, num_samples=1000, anomaly_rate=0.1):
    for i in range(num_sets):
        # Создаем данные без аномалий
        data = np.random.normal(loc=20, scale=5, size=num_samples)

        # Добавляем аномалии
        anomalies = np.random.normal(loc=30, scale=10, size=int(anomaly_rate * num_samples))
        data[:int(anomaly_rate * num_samples)] += anomalies

        # Сохраняем данные в файлы
        file_name = f"data_set_{i}.csv"
        np.savetxt(os.path.join("train", file_name) if i < num_sets // 2 else os.path.join("test", file_name), data,
                   delimiter=",")


if __name__ == "__main__":
    create_data_sets()
