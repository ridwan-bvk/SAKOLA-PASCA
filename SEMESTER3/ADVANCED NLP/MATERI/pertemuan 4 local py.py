import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
# import str

# Pastikan nltk resources sudah diunduh
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------
def preprocess_text(text):
   text = text.lower()
   text = text.translate(text.maketrans("", "", string.punctuation))
   tokens = word_tokenize(text)
   stop_words = set(stopwords.words('english'))
   filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]

   ps = PorterStemmer()
   stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
   lemmatizer = WordNetLemmatizer()
   lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
   pos_tagged_tokens1 = pos_tag(stemmed_tokens)
   pos_tagged_tokens2 = pos_tag(lemmatized_tokens)
   return {
       "original_text": text,
       "tokens": tokens,
       "filtered_tokens": filtered_tokens,
       "stemmed_tokens": stemmed_tokens,
       "lemmatized_tokens": lemmatized_tokens,
       "pos_tagged_tokens1": pos_tagged_tokens1,
       "pos_tagged_tokens2": pos_tagged_tokens2,
 }

sentences = [
 "Saya suka belajar data science.",
 "Python adalah bahasa pemrograman yang populer.",
 "Saya menggunakan Python untuk analisis data.",
 "Analisis data membantu dalam pengambilan keputusan.",
 "Machine learning adalah cabang dari kecerdasan buatan.",
 "Algoritma machine learning bisa memprediksi hasil.",
 "Saya tertarik pada teknologi baru.",
 "Kecerdasan buatan memiliki banyak aplikasi.",
 "Belajar data science sangat menarik.",
 "Saya mengikuti kursus online tentang machine learning."
]

# #####------------------------------

for i, sentence in enumerate(sentences):
    result = preprocess_text(sentence)
    print(f"Hasil preprocessing kalimat {i+1}:\n")
    
    for key, value in result.items():
        print(f"{key}:\n{value}\n")

    print("-" * 50)

# -----------------------------------------

def preprocess_text(text):
     text = text.lower()
     text = text.translate(text.maketrans("", "", string.punctuation))
     text = re.sub(r'\W', ' ', text)
     words = word_tokenize(text)
     stop_words = set(stopwords.words('indonesian'))
     words = [word for word in words if word not in stop_words]
     stemmer = PorterStemmer()
     words = [stemmer.stem(word) for word in words]
     return ' '.join(words)

preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(preprocessed_sentences)

print("Vocabulary:\n", vectorizer.get_feature_names_out())
print('\nVector vocabulary:\n',vectorizer.vocabulary_)
print("\nBag of Words:\n", X_bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_sentences)

print("Vocabulary:\n", tfidf_vectorizer.get_feature_names_out())
print('\nVector vocabulary:\n',tfidf_vectorizer.vocabulary_)
print("\nTF-IDF:\n", X_tfidf.toarray())
