from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset from local folder in Colab
df = pd.read_csv('train.csv') # Ubah 'filename.csv' sesuai dengan nama file Anda

# hilangkan data yang tidak dipakai
df = df.drop(['Name','Ticket','PassengerId','Cabin','Survived','SibSp','Parch','Sex'],axis=1)

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

# print(df.head())
# Mengecek keberadaan nilai NaN dalam DataFrame
# missing_values = df.isna().sum()
# print("Jumlah nilai NaN dalam setiap kolom:")
# print(missing_values)
# print(df)

label = LabelEncoder()
data_column = ['Embarked']
for column in data_column:
    df[column] = label.fit_transform(df[column])


# Mengambil kolom yang bertipe data float
# float_columns = df.select_dtypes(include=['float64']).columns
# for col in float_columns:
#     df[col] = df[col].astype('int32')
# features = df.columns

# df = df.drop(['survived'], 1).astype(float)
features = df
# df.columns.format()
# print(df.dtypes)
print(df.head())
# Normalisasi fitur-fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# # Metode Elbow untuk menentukan jumlah klaster optimal
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


# # Plot inertia vs. jumlah klaster
plt.plot(K, inertia, 'bo-')
plt.xlabel('Jumlah Klaster')
plt.ylabel('Inertia')
plt.title('Metode Elbow untuk Menentukan Jumlah Klaster Optimal')
plt.show()

# # Menggunakan PCA untuk reduksi dimensi
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)

# # Menerapkan K-Means dengan jumlah klaster yang dipilih (misalnya 4)
kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(X_scaled)

# # Plot hasil klastering
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('Hasil Klastering dengan K-Means setelah PCA')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.show()

# # Menampilkan jumlah data dalam setiap klaster
unique, counts = np.unique(y_kmeans, return_counts=True)
print(f'Jumlah data dalam setiap klaster: {dict(zip(unique, counts))}')

# # Menambahkan hasil klastering ke dataframe asli
df['Cluster'] = y_kmeans

# # Membuat plot distribusi fitur berdasarkan klaster
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=feature, data=df)
    plt.title(f'Distribusi {feature} berdasarkan Klaster')
    plt.show()

# # Menghitung dan menampilkan Silhouette Score
    silhouette_avg = silhouette_score(X_scaled, y_kmeans)
    print(f'Silhouette Score untuk klastering: {silhouette_avg}')

# range_n_clusters = list(range(2, 11))

# Menyimpan skor silhouette untuk setiap jumlah klaster
# silhouette_scores = []

# for n_clusters in range_n_clusters:
    # Membuat model KMeans
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Melatih model
    
    
    # Menghitung silhouette score
    # silhouette_avg = silhouette_score(X_scaled, y_kmeans)
    # silhouette_scores.append(silhouette_avg)
    # print(f'Silhouette Score untuk klastering: {silhouette_avg}')
# Menampilkan grafik silhouette score
# plt.figure(figsize=(10, 6))
# plt.plot(range_n_clusters, silhouette_scores, marker='o')
# plt.title('Silhouette Score untuk Berbagai Jumlah Klaster')
# plt.xlabel('Jumlah Klaster')
# plt.ylabel('Silhouette Score')
# plt.xticks(range_n_clusters)
# plt.grid(True)
# plt.show()

