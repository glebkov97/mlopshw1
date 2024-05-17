import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml

# Шаг 1: Импорт необходимых библиотек
# Шаг 2: Загрузка датасета
titanic = fetch_openml('titanic', version=1)
df = pd.DataFrame(titanic.data, columns=titanic.feature_names)

# Шаг 3: Вычисление среднего значения столбца "Age"
age_mean = df['age'].mean()

# Шаг 4: Замена NaN значений на среднее значение
imputer = SimpleImputer(strategy='constant', fill_value=age_mean)
df_imputed = imputer.fit_transform(df[['age']])

# Преобразование обратно в DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=['age'])

# Объединение импутированного столбца "Age" с исходным DataFrame
df_final = pd.concat([df.drop(['age'], axis=1), df_imputed], axis=1)

# Шаг 5: Сохранение обновленного датасета в новый CSV-файл
df_final.to_csv(r'C:\Users\kovj\Desktop\mlopshw1\lab4\modified_titanicAge.csv', index=False)


