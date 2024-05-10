import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import  confusion_matrix,classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, log_loss, matthews_corrcoef
from sklearn import metrics, tree
from sklearn.tree import plot_tree
df = pd.read_csv('train.csv')
# test_data = pd.read_csv('test.csv')
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
y_proba = classifier.predict_proba(X_test)

confusion_matrix = confusion_matrix(y_test,y_pred)
# print(confusion_matrix)
confusion_matrix_prob =pd.crosstab(y_test,y_pred)
print(confusion_matrix_prob)
print('Y predi')
print(y_pred)

ca = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) # Menangani kasus pembagian dengan nol
recall = recall_score(y_test, y_pred, average='weighted')
specificity = recall_score(y_test, y_pred, average='weighted')
logloss = log_loss(y_test, y_proba)
mcc = matthews_corrcoef(y_test, y_pred)
name_classifiers = "Decision tree"
# mencari nilai kurva roc
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
plt.plot(fpr, tpr, label=name_classifiers)
roc_auc = roc_auc_score(y_test, y_proba[:, 1])

# output

# print(f"Model: {name_classifiers}")
# print("Confusion Matrix:")
# print(confusion_matrix)
# print("Accuracy:", accuracy_score)
# print("f1:",f1)
# print("precision:", precision)
# print("recall:", recall)
# print("specificity:", specificity)
print("logloss:", logloss)
# print("mcc:", mcc)
# print("auroc:", roc_auc)

# # confusion_matrix png
# plt.figure(figsize=(10, 5))
# sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='cool')
# plt.show()

# # grafik AOC
# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve - ' + name_classifiers)
# plt.legend(loc="lower right")
# plt.show()

# Generate an image of the tree //pip install pydotplus//pip install graphviz
# from io import StringIO
# from IPython.display import Image, display
# import pydotplus

# out = StringIO()
# tree.export_graphviz(classifier, out_file=out)

# img = pydotplus.graph_from_dot_data(out.getvalue())
# img.write_png('titanic.png')

# plt.figure(figsize=(15,10))
# plot_tree(classifier, feature_names=train_data.columns, class_names=['Not Survived', 'Survived'], filled=True,max_depth=5, fontsize=5)
# plt.show()