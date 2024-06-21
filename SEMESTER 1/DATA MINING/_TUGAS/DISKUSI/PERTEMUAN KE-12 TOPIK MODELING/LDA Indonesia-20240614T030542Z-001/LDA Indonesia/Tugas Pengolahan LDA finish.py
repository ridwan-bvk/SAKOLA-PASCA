import gensim
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
import pandas as pd 
import numpy as np
import nltk
import string 
import re #regex library


# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
# from nltk.probability import FreqDist
from nltk.corpus import stopwords

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = pd.read_excel("dataBerita.xlsx")
data = pd.DataFrame(data)
data = data.drop(columns=["articlename"])

# print(df_imp_wcount)
# pre-processing
# bikin jadi lower semua data
data["textdata"]= data["textdata"].str.lower()
# TOKENISASI
nltk.download('punkt')

def remove_tweet_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    # text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
data['textdata'] = data['textdata'].apply(remove_tweet_special)

#menghilankan number pada text
def remove_number(text):
    return  re.sub(r"\d+", "", text)

data['textdata'] = data['textdata'].apply(remove_number)

#remove tanda baca
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

data['textdata'] = data['textdata'].apply(remove_punctuation)

#remove whitespace 
def remove_whitespace_LT(text):
    return text.strip()

data['textdata'] = data['textdata'].apply(remove_whitespace_LT)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

data['textdata'] = data['textdata'].apply(remove_singl_char)

# NLTK word tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

data['textdata_tokens'] = data['textdata'].apply(word_tokenize_wrapper)

nltk.download('stopwords')

from nltk.corpus import stopwords
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

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

total_topics = 3 # jumlah topik yang akan di extract
number_words = 10 # jumlah kata per topik

# Running and Trainign LDA model on the document term matrix.
lda_model = Lda(doc_term_matrix, num_topics=total_topics, id2word = dictionary, passes=50)

for idx, topic in lda_model.print_topics(-1):
    print(f"Topik {idx}: {topic}")

# Visualisasi dengan pyLDAvis
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis_data)

# Menyimpan visualisasi ke dalam file HTML
pyLDAvis.save_html(vis_data, 'lda_visualization.html')

