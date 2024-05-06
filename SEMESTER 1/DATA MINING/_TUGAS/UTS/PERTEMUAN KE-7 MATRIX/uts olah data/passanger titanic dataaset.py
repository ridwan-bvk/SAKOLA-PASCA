import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, log_loss, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


# https://www.kaggle.com/code/sandhyakrishnan02/logistic-regression-titanic-dataset#20.--Building-Logistic-Regression
## import data
data = pd.read_csv ("passenger titanic dataset.csv")
df =pd.DataFrame(data)
# print(df)

# hilangkan data yang tidak dipakai
df = df.drop(['Name','Ticket','PassengerId','Cabin'],axis=1)

# print(df)
label = LabelEncoder()

# Mengecek keberadaan nilai NaN dalam DataFrame
# missing_values = df.isna().sum()

# Mengganti nilai NaN dalam kolom yg terdapat Nan dengan nilai tertentu untuk setiap kolom
replacement_values = {    'Fare': df['Fare'].median(),}
df.fillna(replacement_values, inplace=True)

# fungsi replace null pada kolom age
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

# missing_values = df.isna().sum()
# Menampilkan jumlah nilai NaN dalam setiap kolom
# print("Jumlah nilai NaN dalam setiap kolom:")
# print(missing_values)

# Menampilkan nama kolom yang memiliki nilai NaN
# print("\nKolom-kolom dengan nilai NaN:")
# print(missing_values[missing_values > 0].index.tolist())


data_column = ['Sex','Embarked']
for column in data_column:
    df[column] = label.fit_transform(df[column])

# tentukan target
X = df.drop(['Survived'],axis = 1)
y = df['Survived']

# buat pembagian dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101)

# buat tools classifiers
classifiers = {
'Logistic Regression': LogisticRegression(max_iter=4000),
'Decision Tree': DecisionTreeClassifier(),
'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=30),
'Naive Bayes': GaussianNB()
}
# declare 
results = {}
roc_fig, roc_ax = plt.subplots()
# pengolahan data setiap classifiers
for name_classifiers, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    # 
    cm = confusion_matrix(y_test, y_pred)
    ca = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) # Menangani kasus pembagian dengan nol
    recall = recall_score(y_test, y_pred, average='weighted')
    specificity = recall_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    # 
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    plt.plot(fpr, tpr, label=name_classifiers)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    results[name_classifiers] = {'confusion_matrix': cm,
                                 'accuracy': ca,
                                 'f1':f1,
                                 'precision':precision,
                                 'recall':recall,
                                 'specificity':specificity,
                                 'logloss':logloss,
                                 'mcc':mcc,
                                 'auroc': roc_auc
                                 }
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ' + name_classifiers)
    plt.legend(loc="lower right")
    plt.show()
# satu curve
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()

for name_classifiers, result in results.items():
    print(f"Model: {name_classifiers}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Accuracy:", result['accuracy'])
    print("f1:", result['f1'])
    print("precision:", result['precision'])
    print("recall:", result['recall'])
    print("specificity:", result['specificity'])
    print("logloss:", result['logloss'])
    print("mcc:", result['mcc'])
    print("auroc:", result['auroc'])
    print()

# for name_classifiers, result in results.items():
#     print(f"Model: {name_classifiers}")
#     print("AUROC:", result['auroc'])
#     print()

    # auroc = roc_auc_score(y_test, y_proba, multi_class='ovr')
# fpr, tpr, _ = roc_curve(y_test, y_proba[:,1], pos_label=1)
# f1 = f1_score(y_test, y_pred, average='weighted')
# precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) # Menangani kasus pembagian dengan nol
# recall = recall_score(y_test, y_pred, average='weighted')
# specificity = recall_score(y_test, y_pred, average='weighted', pos_label=0)
# logloss = log_loss(y_test, y_proba)
# mcc = matthews_corrcoef(y_test, y_pred)
# auroc = roc_auc_score(y_test, y_proba, multi_class='ovr')
# fpr, tpr, _ = roc_curve(y_test, y_proba[:,1], pos_label=1)

# buat dummies nuumerical jang int
# df = pd.get_dummies(df, columns = ['Sex','Embarked'], drop_first=True)
# df = pd.get_dummies(df, columns = ['Embarked'], drop_first=True)
# bersihkan data yag ada null/Nan
# df['Age'].isnull().sum()
# df.isnull().sum()
# Print metrics
# print(f"{name_classifiers}:")
# print("Confusion Matrix:")
# print(cm)
# print("Accuracy:", ca)
# print("F1 Score:", f1)
# print("Precision:", precision)
# print("Recall:", recall)
# print("Specificity:", specificity)
# print("Log Loss:", logloss)
# print("MCC:", mcc)
# print("AUROC:", auroc)
# print(df.head())
# print('x = \n', X.head(),end='\n\n')
# print('y = \n', y.head())
# buat dummy data numeric
# df = pd.get_dummies()