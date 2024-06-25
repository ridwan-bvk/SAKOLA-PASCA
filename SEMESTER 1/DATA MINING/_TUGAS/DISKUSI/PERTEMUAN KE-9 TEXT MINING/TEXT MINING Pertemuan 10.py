

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
# from transformers import (GPT2Config,GPT2LMHeadModel,GPT2Tokenizer)
from string import punctuation as pnc
from collections import Counter
from wordcloud import WordCloud
from math import log

df = pd.read_csv('tweets.csv')
# print(df.head())
# AMBIL 3 KOLOM AJA
df = df[['handle','text','is_retweet']]
# print(df.head())

# Menghitung nilai dari setiap handle
handle_label = df['handle'].value_counts()

# Membuat barplot
sns.barplot(x=handle_label.index, y=handle_label.values)

# Menambahkan label ke sumbu x dan y
plt.xlabel('Handle')
plt.ylabel('Count')

# Menampilkan plot
# plt.show()

# ''' memisahkan donald and hillary tweets '''
realDonaldTrump = df[df.handle == 'realDonaldTrump']
hillaryClinton = df[df.handle == 'HillaryClinton']
print(realDonaldTrump.head())

def get_word_cloud(df, c):
    cm = ' '
    s_word = set(STOPWORDS)
    
    for sent in df[c]:
        # ''' converting sent into string '''
        sent = str(sent)
        # ''' spiltting every sent from (" ") '''
        tokens = sent.split()
        
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        # ''' joining all tokesn '''
        cm += " ".join(tokens)
    
    word_cloud = WordCloud(width=800, height=400, background_color='black', stopwords=s_word,
                           min_font_size=10).generate(cm)
    
    plt.figure(figsize = (10, 10), facecolor = None) 
    plt.imshow(word_cloud) 
    plt.axis("off")
    plt.tight_layout(pad = 0) 
    plt.show()
# untuk hilary clinton
get_word_cloud(hillaryClinton,'text')
# untuk donaldtrump
get_word_cloud(realDonaldTrump, 'text')

# nge-ekstrak kata yang ada @ 
def extract_words(df, c):
    words = []
    for t in df[c].tolist():
        t = [x for x in t.split() if x.startswith('@')]
        words += t
    print(words[:10])

extract_words(realDonaldTrump, 'text')
extract_words(hillaryClinton, 'text')
   
 # nge-ekstrak kata yang ada #   
def extract_words_(df, c):
    words = []
    for t in df[c].tolist():
        t = [x for x in t.split() if x.startswith('#')]
        words += t
    print(words[:10])

extract_words(realDonaldTrump, 'text')
extract_words(hillaryClinton, 'text')

# nge-ekstrak kata yang ada # (—) 
def extract_words_(df, c):
    words = []
    for t in df[c].tolist():
        t = [x for x in t.split() if x.startswith('—')]
        words += t
    
    print(words[:10])
extract_words(realDonaldTrump, 'text')
extract_words(hillaryClinton, 'text')

# ''' buang all tags (@, #, -) '''
def remove_tags(t):
    text = " ".join([x for x in t.split(" ") if not x.startswith("@")])
    text = " ".join([x for x in text.split(" ") if not x.startswith("#")])
    text = " ".join([x for x in text.split(" ") if not x.startswith("—")])
    return text

num_hillary_tweets = len(hillaryClinton)
num_trump_tweets = len(realDonaldTrump)
print(f"Jumlah tweet dari Hillary Clinton: {num_hillary_tweets}")
print(f"Jumlah tweet dari Donald Trump: {num_trump_tweets}")

# Fungsi untuk menghitung kata yang paling sering muncul
def most_common_words(df, column, num_words=10):
    text = " ".join(df[column].tolist())
    words = [word.lower() for word in text.split() if word.lower() not in STOPWORDS and word not in pnc]
    word_counts = Counter(words)
    return word_counts.most_common(num_words)

# Menampilkan 10 kata yang paling sering muncul dalam tweet Hillary Clinton
hillary_common_words = most_common_words(hillaryClinton, 'text')
print(f"Kata yang paling sering muncul dalam tweet Hillary Clinton: {hillary_common_words}")

# Menampilkan 10 kata yang paling sering muncul dalam tweet Donald Trump
trump_common_words = most_common_words(realDonaldTrump, 'text')
print(f"Kata yang paling sering muncul dalam tweet Donald Trump: {trump_common_words}")

# Preprocessing dokumen
def preprocess(document):
    return document.lower().split()

# Tokenisasi dokumen
documents = df
tokenized_documents = [preprocess(doc) for doc in documents]

# Membuat set dari semua kata unik
all_terms = set(term for doc in tokenized_documents for term in doc)

# Menghitung TF (Term Frequency)
def compute_tf(term, document):
    return document.count(term) / len(document)

# Menghitung IDF (Inverse Document Frequency)
def compute_idf(term, all_docs):
    num_docs_with_term = sum(1 for doc in all_docs if term in doc)
    return log(len(all_docs) / (1 + num_docs_with_term))

# Menghitung TF-IDF untuk semua dokumen
def compute_tf_idf(documents):
    tf_idf = []
    for doc in documents:
        doc_tf_idf = {}
        for term in all_terms:
            tf = compute_tf(term, doc)
            idf = compute_idf(term, documents)
            doc_tf_idf[term] = tf * idf
        tf_idf.append(doc_tf_idf)
    return tf_idf

# Memproses dan menghitung TF-IDF
tf_idf_scores = compute_tf_idf(tokenized_documents)

# Menampilkan hasil
for i, doc_tf_idf in enumerate(tf_idf_scores):
    print(f"TF-IDF untuk Dokumen {i+1}:")
    for term, score in doc_tf_idf.items():
        print(f"  {term}: {score:.4f}")
    print()


