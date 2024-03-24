import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import time

# Baca data dari file excel
df = pd.read_excel('D:\ABI\__MATA KULIAH PASCA\SAKOLA-PASCA\SEMESTER 1\DATA MINING\_TUGAS\DISKUSI\PERTEMUAN KE-4 DECISION TREE\heart_disease.xlsx')
df.head()

# Melakukan one-hot encoding pada fitur kategorikal
# data_encoded = pd.get_dummies(data)
# print(data_encoded.columns)

# # Memasukkan kolom-kolom ke dalam fitur (X)
# X = data_encoded[['age', 'rest SBP', 'cholesterol', 'fasting blood sugar > 120', 'max HR',
#                   'exerc ind ang', 'ST by exercise', 'major vessels colored',
#                   'diameter narrowing', 'thal_fixed defect', 'thal_normal',
#                   'thal_reversable defect', 'gender_female', 'gender_male',
#                   'chest pain_asymptomatic', 'chest pain_atypical ang',
#                   'chest pain_non-anginal', 'chest pain_typical ang',
#                   'rest ECG_ST-T abnormal', 'rest ECG_left vent hypertrophy',
#                   'rest ECG_normal', 'slope peak exc ST_downsloping',
#                   'slope peak exc ST_flat', 'slope peak exc ST_upsloping']]

# y = data_encoded['chest pain_typical ang']


# # Membagi data menjadi data latih dan data uji
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Inisialisasi model Decision Tree Classifier
# model = DecisionTreeClassifier()

# # Waktu awal training
# start_train = time.time()

# # Melatih model
# model.fit(X_train, y_train)

# # Waktu akhir training
# end_train = time.time()

# # Waktu training
# train_time = end_train - start_train

# # Waktu awal pengujian
# start_test = time.time()

# # Menggunakan model untuk memprediksi data uji
# y_pred = model.predict(X_test)

# # Waktu akhir pengujian
# end_test = time.time()

# # Waktu pengujian
# test_time = end_test - start_test

# # Menghitung metrik evaluasi
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Menghitung True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN)
# TP = conf_matrix[1, 1]
# TN = conf_matrix[0, 0]
# FP = conf_matrix[0, 1]
# FN = conf_matrix[1, 0]

# # Menghitung Accuracy (CA)
# accuracy_manual = (TP + TN) / (TP + TN + FP + FN)

# # Menghitung Precision
# precision_manual = TP / (TP + FP)

# # Menghitung Recall
# recall_manual = TP / (TP + FN)

# # Menghitung F1 Score
# f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual)

# # Menghitung Specificity
# specificity_manual = TN / (TN + FP)

# # Menghitung MCC (Matthews correlation coefficient)
# mcc_manual = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

# # Menghitung Log Loss dengan penanganan khusus
# y_pred_prob = model.predict_proba(X_test)
# epsilon = 1e-15  # Nilai yang sangat kecil untuk mencegah pembagian dengan nol
# y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)  # Menyamakan probabilitas dengan epsilon dan 1 - epsilon
# logloss_manual = - (1 / len(y_test)) * np.sum(y_test * np.log(y_pred_prob[:,1]) + (1 - y_test) * np.log(1 - y_pred_prob[:,1]))

# # Menampilkan hasil
# print("Confusion Matrix:")
# print(conf_matrix)
# print("Train Time:", train_time)
# print("Test Time:", test_time)
# print("Accuracy (CA):", accuracy_manual)
# print("Precision:", precision_manual)
# print("Recall:", recall_manual)
# print("F1 Score:", f1_manual)
# print("Specificity:", specificity_manual)
# print("MCC (Matthews correlation coefficient):", mcc_manual)
# print("Log Loss:", logloss_manual)
