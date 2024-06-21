import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Contoh data
data = {'text': [
    'I love programming and data science',
    'Python is great for data analysis',
    'I enjoy learning new things',
    'Python programming is fun',
    'Data science is a fascinating field'
]}
df = pd.DataFrame(data)

# Membersihkan teks
def clean_text(text):
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = re.sub(r'\W', ' ', text)  # Menghapus karakter non-alfanumerik
    text = re.sub(r'\s+', ' ', text)  # Menghapus spasi berlebih
    return text.strip()

df['cleaned_text'] = df['text'].apply(clean_text)

# Tokenisasi dan menghapus stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['cleaned_text'].apply(preprocess_text)

# Menggunakan CountVectorizer untuk menghitung frekuensi kata
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Mendapatkan fitur (kata)
feature_names = vectorizer.get_feature_names_out()

# Mengubah vektor menjadi array
X_array = X.toarray()

# Menghitung total frekuensi kata
word_freq = X_array.sum(axis=0)

# Membuat DataFrame dari frekuensi kata
word_freq_df = pd.DataFrame({'word': feature_names, 'frequency': word_freq})

# Mengurutkan berdasarkan frekuensi
word_freq_df = word_freq_df.sort_values(by='frequency', ascending=False)

# Mencetak kata-kata dengan frekuensi tertinggi
print(word_freq_df)

# Menampilkan 10 kata paling sering muncul
print("\nTop 10 words by frequency:")
print(word_freq_df.head(10))
