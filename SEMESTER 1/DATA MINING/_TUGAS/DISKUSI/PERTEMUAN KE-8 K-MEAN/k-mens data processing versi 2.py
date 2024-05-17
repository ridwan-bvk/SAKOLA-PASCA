import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat dataset CSV ke dalam dataframe
df = pd.read_csv("train.csv") # Ganti 'nama_file.csv' dengan nama file CSV yang sudah ada di Google Colab


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
# df = df.drop(['Pclass'],axis=1)
label = LabelEncoder()
data_column = ['Embarked']
for column in data_column:
    df[column] = label.fit_transform(df[column])
    
# Menampilkan lima baris pertama dari dataframe
# print("Lima Baris Pertama Data:")
# print(df.head())
# Mengambil kolom yang bertipe data float

float_columns = df.select_dtypes(include=['float64']).columns
for col in float_columns:
    df[col] = df[col].astype('int32')

# features = df.columns
# print(df.dtypes)
# print(df.head())
# Memilih fitur yang akan digunakan untuk clustering
# 
scaler = StandardScaler()
df = scaler.fit_transform(df)

X = df.columns
# Mencari jumlah klaster yang optimal menggunakan metode Elbow
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Metode Elbow untuk Menentukan Jumlah Klaster Optimal')
plt.xlabel('Jumlah Klaster')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# # Mencari jumlah klaster yang optimal menggunakan metode Silhouette
# silhouette_scores = []
# for n_cluster in range(2, 11):
#     silhouette_scores.append(
#     silhouette_score(X, KMeans(n_clusters = n_cluster).fit_predict(X)))

# # Plotting Silhouette Score
#     k = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#     plt.bar(k, silhouette_scores)
#     plt.xlabel('Jumlah Klaster')
#     plt.ylabel('Silhouette Score')
#     plt.title('Metode Silhouette untuk Menentukan Jumlah Klaster Optimal')
#     plt.show()

# # Menentukan jumlah klaster optimal berdasarkan hasil Elbow atau Silhouette
# # Misalnya, kita pilih jumlah klaster = 3
# n_clusters = 3

# # Menerapkan algoritma K-Means dengan jumlah klaster optimal
# kmeans = KMeans(n_clusters=n_clusters)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)

# # Menambahkan kolom ClustID ke dalam dataframe
# df['ClustID'] = y_kmeans

# # Visualisasi hasil clustering
# plt.scatter(X['num_reactions'], X['num_comments'], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
# plt.xlabel('Number of Reactions')
# plt.ylabel('Number of Comments')
# plt.title('K-Means Clustering')
# plt.show()

# # Menampilkan hasil klaster K-Means
# print("\nHasil Klaster K-Means:")
# print(df)
