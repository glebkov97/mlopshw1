from sklearn.datasets import fetch_openml
import pandas as pd

# Загрузка датасета Titanic
titanic = fetch_openml('titanic', version=1)
df = pd.DataFrame(titanic.data, columns=titanic.feature_names)

# Выбор нужных столбцов и преобразование их
modified_df = df[['pclass', 'sex', 'age']].copy()
modified_df['Class'] = modified_df['pclass'].apply(lambda x: 'First' if x == 1 else ('Second' if x == 2 else 'Third'))

modified_df.to_csv(r'C:\Users\kovj\Desktop\mlopshw1\lab4\modified_titanic.csv', index=False)
