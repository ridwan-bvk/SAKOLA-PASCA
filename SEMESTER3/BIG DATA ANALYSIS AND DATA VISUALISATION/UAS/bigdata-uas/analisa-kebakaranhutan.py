# %% [1] IMPORT LIBRARY DAN LOAD DATA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer

# Load dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
url ="E:\\pythonAppGado\\bigdata-uas\\forestfires.csv"
data = pd.read_csv(url)

# %% [2] PREPROCESSING DATA
# 2.1. Pemeriksaan data awal
print("=== Info Dataset ===")
print(data.info())
print("\n=== Statistik Deskriptif ===")
print(data.describe())

# 2.2. Penanganan nilai hilang
print("\n=== Nilai Hilang Sebelum Penanganan ===")
print(data.isnull().sum())

imputer = SimpleImputer(strategy='median')
data[['rain']] = imputer.fit_transform(data[['rain']])

# 2.3. Transformasi variabel target (area > 0 = kebakaran)
data['fire'] = np.where(data['area'] > 0, 1, 0)

# 2.4. Seleksi fitur
features = ['temp', 'RH', 'wind', 'rain']
target = 'fire'
X = data[features]
y = data[target]

# 2.5. Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2.6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# %% [3] VISUALISASI DATA
plt.figure(figsize=(15, 12))

# 3.1. Distribusi kelas target
plt.subplot(2, 2, 1)
sns.countplot(x='fire', data=data)
plt.title('Distribusi Kejadian Kebakaran (0: Tidak, 1: Ya)')
plt.xlabel('Status Kebakaran')
plt.ylabel('Jumlah')

# 3.2. Distribusi fitur suhu
plt.subplot(2, 2, 2)
sns.histplot(data=data, x='temp', hue='fire', kde=True, element='step', palette='viridis')
plt.title('Distribusi Suhu vs Kebakaran')
plt.xlabel('Suhu (°C)')

# 3.3. Distribusi kelembaban
plt.subplot(2, 2, 3)
sns.boxplot(x='fire', y='RH', data=data, palette='coolwarm')
plt.title('Kelembaban Relatif vs Kebakaran')
plt.xlabel('Status Kebakaran')
plt.ylabel('Kelembaban (%)')

# 3.4. Korelasi fitur
plt.subplot(2, 2, 4)
corr_matrix = data[['temp', 'RH', 'wind', 'rain', 'fire']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriks Korelasi Fitur')

plt.tight_layout()
plt.savefig('forest_fires_visualization.png', dpi=300)
plt.show()

# %% [4] PEMODELAN K-NEAREST NEIGHBORS
# 4.1. Hyperparameter tuning dengan GridSearchCV
param_grid = {
    'n_neighbors': range(3, 15),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 4.2. Model terbaik
best_knn = grid_search.best_estimator_
print("\n=== Parameter Terbaik ===")
print(grid_search.best_params_)

# 4.3. Evaluasi model
y_pred = best_knn.predict(X_test)
y_prob = best_knn.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 4.4. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Tidak', 'Ya'], 
            yticklabels=['Tidak', 'Ya'])
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# 4.5. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png', dpi=300)
plt.show()

# %% [5] ANALISIS FITUR PENTING
# Menghitung kepentingan fitur berdasarkan akurasi
feature_importance = []
for i in range(len(features)):
    X_temp = np.delete(X_train, i, axis=1)
    knn_temp = KNeighborsClassifier(**grid_search.best_params_)
    knn_temp.fit(X_temp, y_train)
    score = knn_temp.score(np.delete(X_test, i, axis=1), y_test)
    feature_importance.append(score)

# Visualisasi kepentingan fitur
plt.figure(figsize=(10, 6))
sns.barplot(x=[features[i] for i in range(len(features))], 
            y=feature_importance, palette="viridis")
plt.axhline(y=best_knn.score(X_test, y_test), color='r', linestyle='--')
plt.title('Pengaruh Penghapusan Fitur Terhadap Akurasi')
plt.ylabel('Akurasi')
plt.xlabel('Fitur yang Dihapus')
plt.savefig('feature_importance.png', dpi=300)
plt.show()