# %% [1] Import Library
import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# %% [2] Load Dataset
def load_data(file_path, sample_size=10000):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df.sample(min(sample_size, len(df))), df

df_sample, df_full = load_data('E:\\pythonAppGado\\bigdata-uas\\archive\\News_Category_Dataset_v3.json', 15000)

# %% [3] Preprocessing Data
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(df):
    # Handle missing values
    df = df.dropna(subset=['headline', 'short_description'])
    
    # Combine text columns
    df['combined_text'] = df['headline'] + ' ' + df['short_description']
    
    # Clean text
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def process_text(text):
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)
    
    df['processed_text'] = df['cleaned_text'].apply(process_text)
    return df

df_processed = preprocess_data(df_sample.copy())

# %% [4] TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.7
)

tfidf_matrix = tfidf_vectorizer.fit_transform(df_processed['processed_text'])
feature_names = tfidf_vectorizer.get_feature_names_out()

# %% [5] Clustering
# K-Means Clustering
kmeans = KMeans(
    n_clusters=8,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)
kmeans.fit(tfidf_matrix)
df_processed['kmeans_cluster'] = kmeans.labels_

# DBSCAN Clustering
dbscan = DBSCAN(
    eps=0.5,
    min_samples=5,
    metric='cosine'
)
dbscan.fit(tfidf_matrix)
df_processed['dbscan_cluster'] = dbscan.labels_

# Evaluate clusters
print(f"K-Means Silhouette Score: {silhouette_score(tfidf_matrix, kmeans.labels_):.3f}")
print(f"DBSCAN Clusters: {len(np.unique(dbscan.labels_))}")

# %% [6] Visualisasi
# Dimensionality Reduction
pca = PCA(n_components=2, random_state=42)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

tfidf_2d_pca = pca.fit_transform(tfidf_matrix.toarray())
tfidf_2d_tsne = tsne.fit_transform(tfidf_matrix.toarray())

# Plot K-Means Clusters
plt.figure(figsize=(15, 10))
scatter = plt.scatter(
    tfidf_2d_tsne[:, 0], 
    tfidf_2d_tsne[:, 1],
    c=df_processed['kmeans_cluster'],
    cmap='viridis',
    alpha=0.6
)
plt.title('K-Means Clustering (t-SNE Visualization)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(scatter, label='Cluster')
plt.savefig('kmeans_clusters.png', dpi=300)
plt.show()

# Plot DBSCAN Clusters
plt.figure(figsize=(15, 10))
scatter = plt.scatter(
    tfidf_2d_tsne[:, 0], 
    tfidf_2d_tsne[:, 1],
    c=df_processed['dbscan_cluster'],
    cmap='plasma',
    alpha=0.6
)
plt.title('DBSCAN Clustering (t-SNE Visualization)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(scatter, label='Cluster')
plt.savefig('dbscan_clusters.png', dpi=300)
plt.show()

# %% [7] Analisis Kata Kunci
def get_top_keywords(cluster_df, tfidf_matrix, n_keywords=15):
    cluster_keywords = {}
    for cluster_id in sorted(cluster_df['kmeans_cluster'].unique()):
        # Get indices of documents in the cluster
        cluster_indices = cluster_df[cluster_df['kmeans_cluster'] == cluster_id].index
        
        # Calculate mean TF-IDF scores for the cluster
        cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
        
        # Get top keywords indices
        top_indices = cluster_tfidf.argsort()[-n_keywords:][::-1]
        
        # Map indices to feature names
        keywords = [feature_names[i] for i in top_indices]
        cluster_keywords[cluster_id] = keywords
    
    return cluster_keywords

top_keywords = get_top_keywords(df_processed, tfidf_matrix)

# Visualize keywords
plt.figure(figsize=(18, 12))
for cluster_id, keywords in top_keywords.items():
    plt.subplot(3, 3, cluster_id + 1)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(' '.join(keywords))
    
    plt.imshow(wordcloud)
    plt.title(f'Cluster {cluster_id} Keywords')
    plt.axis('off')

plt.tight_layout()
plt.savefig('cluster_keywords.png', dpi=300)
plt.show()

# %% [8] Ekspor Hasil Analisis
# Simpan hasil clustering
df_processed[['headline', 'short_description', 'kmeans_cluster']].to_csv('clustered_news.csv', index=False)

# Buat laporan ringkas
with open('cluster_analysis.txt', 'w') as f:
    f.write("Analisis Cluster Berita\n=======================\n\n")
    f.write(f"Jumlah Cluster: {len(top_keywords)}\n")
    f.write(f"Silhouette Score: {silhouette_score(tfidf_matrix, kmeans.labels_):.3f}\n\n")
    
    for cluster_id, keywords in top_keywords.items():
        f.write(f"Cluster {cluster_id}:\n")
        f.write(f"Top Keywords: {', '.join(keywords[:10])}\n")
        f.write(f"Jumlah Artikel: {len(df_processed[df_processed['kmeans_cluster'] == cluster_id])}\n")
        f.write("-" * 50 + "\n")

# %% [9] Analisis Distribusi Cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='kmeans_cluster', data=df_processed, palette='viridis')
plt.title('Distribusi Artikel per Cluster')
plt.xlabel('Cluster ID')
plt.ylabel('Jumlah Artikel')
plt.savefig('cluster_distribution.png', dpi=300)
plt.show()