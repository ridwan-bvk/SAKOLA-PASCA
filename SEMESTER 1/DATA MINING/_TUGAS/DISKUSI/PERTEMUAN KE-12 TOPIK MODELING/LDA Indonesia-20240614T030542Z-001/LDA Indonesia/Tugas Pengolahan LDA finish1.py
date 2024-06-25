import pandas as pd
# import numpy as np
import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import gensim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = pd.read_excel("dataBerita.xlsx")
data = pd.DataFrame(data)
data = data.drop(columns=["articlename"])

# Preprocessing functions (sama seperti sebelumnya)
def remove_tweet_special(text):
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://", " ").replace("https://", " ")

def remove_number(text):
    return re.sub(r"\d+", "", text)

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_whitespace_LT(text):
    return text.strip()

def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

def word_tokenize_wrapper(text):
    return word_tokenize(text)

def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

# Memuat data
data = pd.read_excel("dataBerita.xlsx")
data = pd.DataFrame(data)
data = data.drop(columns=["articlename"])

# Pre-processing
data["textdata"] = data["textdata"].str.lower()
data['textdata'] = data['textdata'].apply(remove_tweet_special)
data['textdata'] = data['textdata'].apply(remove_number)
data['textdata'] = data['textdata'].apply(remove_punctuation)
data['textdata'] = data['textdata'].apply(remove_whitespace_LT)
data['textdata'] = data['textdata'].apply(remove_singl_char)
data['textdata_tokens'] = data['textdata'].apply(word_tokenize_wrapper)

nltk.download('stopwords')
list_stopwords = stopwords.words('indonesian')

list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo',
                       'kalo', 'amp', 'biar', 'bikin', 'bilang',
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', 'bisnis', 'pandemi', 'indonesia',
                       "ada", "tan", "ton", "pt", "komentar", "juta", "unit", "menang", "artikel",
                       "smartphone", "tagar", "sedia", "kaskus", "seksi"])

list_stopwords = set(list_stopwords)
data['textdata_tokens_WSW'] = data['textdata_tokens'].apply(stopwords_removal)

# Menggabungkan semua kata menjadi satu string untuk wordcloud
all_words = ' '.join([' '.join(tokens) for tokens in data['textdata_tokens_WSW']])

# Membuat wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

# Menampilkan wordcloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# convert list to dictionary
list_stopwords = set(list_stopwords)

#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]
data['textdata_tokens_WSW'] = data['textdata_tokens'].apply(stopwords_removal) 

from gensim import corpora
doc_clean = data['textdata_tokens_WSW']
dictionary = corpora.Dictionary(doc_clean)
corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
# print(dictionary)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# buat lda
Lda = gensim.models.ldamodel.LdaModel

total_topics = 2 # jumlah topik yang akan di extract
number_words = 10 # jumlah kata per topik
# Running a LDA model on the document term matrix.
lda_model = Lda(doc_term_matrix, num_topics=total_topics, id2word = dictionary, passes=50)

for idx, topic in lda_model.print_topics(-1):
    print(f"Topik {idx}: {topic}")

# Visualisasi dengan pyLDAvis
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis_data)

# Menyimpan visualisasi ke dalam file HTML
pyLDAvis.save_html(vis_data, 'lda_visualization.html')
