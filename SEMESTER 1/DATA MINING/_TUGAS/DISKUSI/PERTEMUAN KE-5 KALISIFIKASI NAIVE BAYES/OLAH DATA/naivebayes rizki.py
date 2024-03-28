import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,log_loss
import time

# Load data from Excel file
excel_file = 'breastcancer.xlsx'
df = pd.read_excel(excel_file)

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Naive Bayes model
model = GaussianNB()

# Waktu awal training
start_train = time.time()

# Train the model
model.fit(X_train, y_train)

# Waktu akhir training
end_train = time.time()

# Waktu training
train_time = end_train - start_train

# Waktu awal pengujian
start_test = time.time()

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Waktu akhir pengujian
end_test = time.time()

# Waktu pengujian
test_time = end_test - start_test

# Menghitung metrik evaluasi
conf_matrix = confusion_matrix(y_test, y_pred)

# Menghitung True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN)
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

# Menghitung Accuracy (CA)
accuracy_manual = (TP + TN) / (TP + TN + FP + FN)

# Menghitung Precision
precision_manual = TP / (TP + FP)

# Menghitung Recall
recall_manual = TP / (TP + FN)

# Menghitung F1 Score
f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual)

# Menghitung Specificity
specificity_manual = TN / (TN + FP)

# Menghitung MCC (Matthews correlation coefficient)
mcc_manual = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

# Menghitung log loss
y_pred_proba = model.predict_proba(X_test)
logloss = log_loss(y_test, y_pred_proba)

# Menampilkan hasil
print("Confusion Matrix:")
print(conf_matrix)
print("Distribusi kelas dalam data uji:")
print(y_test.value_counts())
print("Train Time:", train_time)
print("Test Time:", test_time)
print("Accuracy (CA):", accuracy_manual)
print("Precision:", precision_manual)
print("Recall:", recall_manual)
print("F1 Score:", f1_manual)
print("Specificity:", specificity_manual)
print("MCC (Matthews correlation coefficient):", mcc_manual)
print("Log Loss:", logloss)

# Plotting Scarlet Plot
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_pred, palette='deep')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Scarlet Plot')
plt.show()