

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from transformers import (GPT2Config,GPT2LMHeadModel,GPT2Tokenizer)
from string import punctuation as pnc
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, log_loss, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB

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

# Preprocessing text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['text'])

# Split features and target variable
X = X_tfidf
y = df['handle'] # Ganti 'target_column' dengan nama kolom target Anda

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize Naive Bayes classifier
clf = GaussianNB()

# Train and evaluate the model
# Convert sparse matrix to dense since GaussianNB does not support sparse matrix
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

clf.fit(X_train_dense, y_train)
y_pred = clf.predict(X_test_dense)
y_proba = clf.predict_proba(X_test_dense)

# Calculate metrics
cm = confusion_matrix(y_test, y_pred)
ca = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) # Handle division by zero case
recall = recall_score(y_test, y_pred, average='weighted')
logloss = log_loss(y_test, y_proba)
mcc = matthews_corrcoef(y_test, y_pred)
auroc = roc_auc_score(y_test, y_proba, multi_class='ovr')

# Print metrics
print("Naive Bayes:")
print("Confusion Matrix:")
print(cm)
print("Accuracy:", ca)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Log Loss:", logloss)
print("MCC:", mcc)
print("AUROC:", auroc)
