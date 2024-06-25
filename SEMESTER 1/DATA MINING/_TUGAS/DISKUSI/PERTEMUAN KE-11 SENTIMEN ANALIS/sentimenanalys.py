import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from wordcloud import WordCloud

# Download NLTK (natural language toolkit) data
# nltk.download('punkt')
# nltk.download('stopwords')

df = pd.read_csv('stock_data.csv') 
# print(df.head())

# Fungsi Membersihkan teks
def clean_text(text):
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = re.sub(r'\W', ' ', text)  # Menghapus karakter non-alfanumerik
    text = re.sub(r'\s+', ' ', text)  # Menghapus spasi berlebih
    text = re.sub(r'\bhttps\b', '', text)  # Menghapus kata 'bhttps'
    text = re.sub(r'\bco\b', '', text)  # Menghapus kata 'com'
    text = re.sub(r'\blike\b', '', text)  # Menghapus kata 'com'
    text = re.sub(r'\baap\b', '', text)  # Menghapus kata 'com'
    text = re.sub(r'\buser\b', '', text)  # Menghapus kata 'com'
    return text.strip()

df['cleaned_text'] = df['Text'].apply(clean_text)
# print (df.head())

# Tokenisasi dan menghapus stopwords
stop_words = set(stopwords.words('english'))

def preproses_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['preproses_text']=df['cleaned_text'].apply(preproses_text)
# print(df.head)

# Menggabungkan semua teks yang telah diproses menjadi satu string
text = ' '.join(df['preproses_text'])

# Menyiapkan data untuk training dan testing
X = df['preproses_text']
y = df['Sentiment']  # Pastikan kolom 'sentiment' sesuai dengan data Anda

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("Data Split into Train and Test!")

# Vektorisasi teks
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# print("Text Vectorization Done!")

# Melatih model
model = LogisticRegression()
model.fit(X_train_vec, y_train)
print("Model Training Done!")

# Memprediksi hasil
y_pred = model.predict(X_test_vec)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Membuat word cloud
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)

# Menampilkan word cloud menggunakan matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# 
X_counts = vectorizer.fit_transform(df['preproses_text'])

# Mendapatkan fitur (kata)
feature_names = vectorizer.get_feature_names_out()

# Menghitung total frekuensi kata
word_freq = X_counts.toarray().sum(axis=0)

# Membuat DataFrame dari frekuensi kata
word_freq_df = pd.DataFrame({'word': feature_names, 'frequency': word_freq})

# Mengurutkan berdasarkan frekuensi
word_freq_df = word_freq_df.sort_values(by='frequency', ascending=False)

# Mencetak kata-kata dengan frekuensi tertinggi
print("Word Frequencies:")
print(word_freq_df.head(10))


sentiment_counts = df['Sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
plt.show()