import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

df = pd.read_csv(r'D:\ABI\__MATA KULIAH PASCA\clone 2\SAKOLA-PASCA\SEMESTER3\BIG DATA ANALYSIS AND DATA VISUALISATION\TUGAS TERSTRUKTUR\Titanic-Dataset.csv')

# print(df.head())

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')


# Tangani nilai yang hilang
# Untuk kolom numerik, isi dengan median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
imputer_num = SimpleImputer(strategy='median')
df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Untuk kolom kategorikal, isi dengan modus (nilai yang paling sering muncul)
cat_cols = df.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Label encoding untuk kolom kategorikal
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Fitur dan label
X = df.drop('Survived', axis=1)
y = df['Survived']

# Normalisasi fitur numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bagi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Buat dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]

# Evaluasi
print("\nAkurasi:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Selamat', 'Selamat'], yticklabels=['Tidak Selamat', 'Selamat'])
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.tight_layout()
plt.savefig("D:/ABI/__MATA KULIAH PASCA/clone 2/SAKOLA-PASCA/SEMESTER3/BIG DATA ANALYSIS AND DATA VISUALISATION/TUGAS TERSTRUKTUR/confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("D:/ABI/__MATA KULIAH PASCA/clone 2/SAKOLA-PASCA/SEMESTER3/BIG DATA ANALYSIS AND DATA VISUALISATION/TUGAS TERSTRUKTUR/roc_curve.png")
plt.close()