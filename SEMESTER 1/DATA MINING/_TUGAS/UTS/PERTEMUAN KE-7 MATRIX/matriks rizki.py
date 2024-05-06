import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, log_loss, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load data from Excel
data = pd.read_excel("C:\Python312\heart_disease.xlsx")

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()

# Mengonversi kolom kategori menjadi angka
categorical_columns = ['gender','thal', 'rest ECG', 'slope peak exc ST', 'chest pain']
for column in categorical_columns:
data[column] = label_encoder.fit_transform(data[column])

# Split features and target variable
X = data.drop(columns=['thal'])
y = data['thal']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize classifiers
classifiers = {
'Logistic Regression': LogisticRegression(max_iter=10000),
'Decision Tree': DecisionTreeClassifier(),
'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=30),
'Naive Bayes': GaussianNB()
}

# Train and evaluate models
for name, clf in classifiers.items():
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Calculate metrics
cm = confusion_matrix(y_test, y_pred)
ca = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) # Menangani kasus pembagian dengan nol
recall = recall_score(y_test, y_pred, average='weighted')
specificity = recall_score(y_test, y_pred, average='weighted', pos_label=0)
logloss = log_loss(y_test, y_proba)
mcc = matthews_corrcoef(y_test, y_pred)
auroc = roc_auc_score(y_test, y_proba, multi_class='ovr')
fpr, tpr, _ = roc_curve(y_test, y_proba[:,1], pos_label=1)

# Print metrics
print(f"{name}:")
print("Confusion Matrix:")
print(cm)
print("Accuracy:", ca)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("Log Loss:", logloss)
print("MCC:", mcc)
print("AUROC:", auroc)

# Plot AUROC Curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()