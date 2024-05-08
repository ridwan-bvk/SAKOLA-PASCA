import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model  import LogisticRegression

from sklearn.metrics import  confusion_matrix,classification_report
from sklearn import metrics, tree

df = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# print(df.head())

# menangani data age 
def impute_train_age(cols):
    Age = cols.iloc[0]
    Pclass = cols.iloc[1]
    
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

train_data = df
train_data.drop(['Cabin'],axis=1,inplace=True)
train_data.dropna(inplace=True)

train_data = pd.get_dummies(train_data, columns = ['Sex'], drop_first=True)
train_data = pd.get_dummies(train_data,columns=['Embarked'],drop_first= True)

train_data.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)

X = train_data.drop(['Survived'],axis = 1)
y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)

# Generate an image of the tree //pip install pydotplus//pip install graphviz
# from io import StringIO
# from IPython.display import Image, display
# import pydotplus

# out = StringIO()
# tree.export_graphviz(classifier, out_file=out)

# img = pydotplus.graph_from_dot_data(out.getvalue())
# img.write_png('titanic.png')