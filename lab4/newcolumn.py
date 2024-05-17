import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml

titanic = fetch_openml('titanic', version=1)
df = pd.DataFrame(titanic.data, columns=titanic.feature_names)

# Преобразование 'Sex' в one-hot encoding
df_encoded = pd.get_dummies(df, columns=['sex'])

# Сохранение преобразованного DataFrame в файл CSV
df_encoded.to_csv(r'C:\Users\kovj\Desktop\mlopshw1\lab4\modified_titanicv3.csv', index=False)
