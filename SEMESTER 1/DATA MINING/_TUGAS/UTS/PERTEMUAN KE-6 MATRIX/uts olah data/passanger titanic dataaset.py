import numpy as np 
import pandas as pd
# https://www.kaggle.com/code/sandhyakrishnan02/logistic-regression-titanic-dataset#20.--Building-Logistic-Regression
## import data
df = pd.read_csv ("passenger titanic dataset.csv")
# print(df)

# hilangkan data yang tidak dipakai
df = df.drop(['Name','Ticket','PassengerId','Cabin'],axis=1)
# print(df)

# fungsi replace null
def impute_train_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    
df['Age'] = df[['Age','Pclass']].apply(impute_train_age,axis=1)
# buat dummies nuumerical jang int
df = pd.get_dummies(df, columns = ['Sex'], drop_first=True)
df = pd.get_dummies(df, columns = ['Embarked'], drop_first=True)
# bersihkan data yag ada null/Nan
# df['Age'].isnull().sum()
# df.isnull().sum()
print(df)
# buat dummy data numeric
# df = pd.get_dummies()