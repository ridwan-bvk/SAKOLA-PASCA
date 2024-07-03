<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction import text

# Mengambil stop words bahasa Indonesia dari Sastrawi
stopword_factory = StopWordRemoverFactory()
stopwords_sastrawi = stopword_factory.get_stop_words()

# Stop words bahasa Inggris dari sklearn
stopwords_sklearn = text.ENGLISH_STOP_WORDS

# Menggabungkan stop words dari kedua sumber
stopwords = list(stopwords_sastrawi) + list(stopwords_sklearn)

# 1. Membaca data
data = pd.read_excel("D:/doc/data mining/Program/data/book-excerpts.xlsx")
texts = data['Text']

# 2. TF-IDF Vectorization dengan stop words
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, max_df=0.8, min_df=5)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# 3. Reduksi dimensi dengan PCA
pca = PCA(n_components=50)  # Ubah jumlah komponen sesuai kebutuhan
X_pca = pca.fit_transform(X_tfidf.toarray())

# 4. Menggunakan SBERT untuk mendapatkan embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(texts)

# 5. Reduksi dimensi dengan UMAP
umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)

# 6a. Clustering dengan KMeans
num_clusters = 5  # Tentukan jumlah kluster
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(umap_embeddings)
kmeans_labels = kmeans.labels_

# 6b. Clustering dengan HDBSCAN
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1)
hdbscan_labels = hdbscan_model.fit_predict(umap_embeddings)

# 7. Menambahkan label kluster ke data asli
data['KMeans_Cluster'] = kmeans_labels
data['HDBSCAN_Cluster'] = hdbscan_labels

# 8. Visualisasi hasil clustering KMeans
plt.figure(figsize=(10, 7))
palette = sns.color_palette("hsv", num_clusters)
sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], hue=kmeans_labels, legend='full', palette=palette)
plt.title('Clustering of Texts using SBERT Embeddings, UMAP, and KMeans')
plt.show()

# 9. Visualisasi hasil clustering HDBSCAN
plt.figure(figsize=(10, 7))
palette = sns.color_palette("hsv", len(set(hdbscan_labels)))
sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], hue=hdbscan_labels, legend='full', palette=palette)
plt.title('Clustering of Texts using SBERT Embeddings, UMAP, and HDBSCAN')
plt.show()

# 10. Menyimpan hasil clustering ke Excel
output_file = 'hasil_clustering.xlsx'
with pd.ExcelWriter(output_file) as writer:
    data.to_excel(writer, sheet_name='Clustering Results', index=False)

print(f'Hasil clustering disimpan di "{output_file}"')
=======
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction import text

# Mengambil stop words bahasa Indonesia dari Sastrawi
stopword_factory = StopWordRemoverFactory()
stopwords_sastrawi = stopword_factory.get_stop_words()

# Stop words bahasa Inggris dari sklearn
stopwords_sklearn = text.ENGLISH_STOP_WORDS

# Menggabungkan stop words dari kedua sumber
stopwords = list(stopwords_sastrawi) + list(stopwords_sklearn)

# 1. Membaca data
data = pd.read_excel("D:/doc/data mining/Program/data/book-excerpts.xlsx")
texts = data['Text']

# 2. TF-IDF Vectorization dengan stop words
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, max_df=0.8, min_df=5)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# 3. Reduksi dimensi dengan PCA
pca = PCA(n_components=50)  # Ubah jumlah komponen sesuai kebutuhan
X_pca = pca.fit_transform(X_tfidf.toarray())

# 4. Menggunakan SBERT untuk mendapatkan embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(texts)

# 5. Reduksi dimensi dengan UMAP
umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)

# 6a. Clustering dengan KMeans
num_clusters = 5  # Tentukan jumlah kluster
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(umap_embeddings)
kmeans_labels = kmeans.labels_

# 6b. Clustering dengan HDBSCAN
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1)
hdbscan_labels = hdbscan_model.fit_predict(umap_embeddings)

# 7. Menambahkan label kluster ke data asli
data['KMeans_Cluster'] = kmeans_labels
data['HDBSCAN_Cluster'] = hdbscan_labels

# 8. Visualisasi hasil clustering KMeans
plt.figure(figsize=(10, 7))
palette = sns.color_palette("hsv", num_clusters)
sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], hue=kmeans_labels, legend='full', palette=palette)
plt.title('Clustering of Texts using SBERT Embeddings, UMAP, and KMeans')
plt.show()

# 9. Visualisasi hasil clustering HDBSCAN
plt.figure(figsize=(10, 7))
palette = sns.color_palette("hsv", len(set(hdbscan_labels)))
sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], hue=hdbscan_labels, legend='full', palette=palette)
plt.title('Clustering of Texts using SBERT Embeddings, UMAP, and HDBSCAN')
plt.show()

# 10. Menyimpan hasil clustering ke Excel
output_file = 'hasil_clustering.xlsx'
with pd.ExcelWriter(output_file) as writer:
    data.to_excel(writer, sheet_name='Clustering Results', index=False)

print(f'Hasil clustering disimpan di "{output_file}"')
>>>>>>> 47b1455b20e230ad6ec1ef7cddb44db890096d47
