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

#remove multiple whitespace into single whitespace
# def remove_whitespace_multiple(text):
#     return re.sub('\s+',' ',text)

# data['textdata'] = data['textdata'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

data['textdata'] = data['textdata'].apply(remove_singl_char)

# NLTK word tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

data['textdata_tokens'] = data['textdata'].apply(word_tokenize_wrapper)

# print('Tokenizing Result : \n') 
# print(data['textdata_tokens'].head())

# NLTK calc frequency distribution
# def freqDist_wrapper(text):
#     return FreqDist(text)

# data['textdata_tokens_fdist'] = data['textdata_tokens'].apply(freqDist_wrapper)

# print('Frequency Tokens : \n') 
# print(data['textdata_tokens_fdist'].head().apply(lambda x : x.most_common()))

nltk.download('stopwords')

from nltk.corpus import stopwords
list_stopwords = stopwords.words('indonesian')


# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
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

# print(data['textdata_tokens_WSW'].head())

# normalizad_word = pd.read_excel('normalisasi.xlsx') #lokasi file

# normalizad_word_dict = {}

# for index, row in normalizad_word.iterrows():
#     if row[0] not in normalizad_word_dict:
#         normalizad_word_dict[row[0]] = row[1] 

# def normalized_term(document):
#     return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

# data['textdata_normalized'] = data['textdata_tokens_WSW'].apply(normalized_term)

# # print(data['textdata_normalized'].head(10))

# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# import swifter

# # create stemmer
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

# # stemmed
# def stemmed_wrapper(term):
#     return stemmer.stem(term)

# term_dict = {}

# for document in data['textdata_tokens_WSW']:
#     for term in document:
#         if term not in term_dict:
#             term_dict[term] = ' '
            
# # print(len(term_dict))
# for term in term_dict:
#     term_dict[term] = stemmed_wrapper(term)

# # apply stemmed term to dataframe
# def get_stemmed_term(document):
#     return [term_dict[term] for term in document]

# data['textdata_tokens_stemmed'] = data['textdata_tokens_WSW'].swifter.apply(get_stemmed_term)
# import gensim
from gensim import corpora
doc_clean = data['textdata_tokens_WSW']
dictionary = corpora.Dictionary(doc_clean)
corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
# print(dictionary)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
# Menampilkan hasil

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

total_topics = 3 # jumlah topik yang akan di extract
number_words = 10 # jumlah kata per topik

# Running and Trainign LDA model on the document term matrix.
lda_model = Lda(doc_term_matrix, num_topics=total_topics, id2word = dictionary, passes=50)

for idx, topic in lda_model.print_topics(-1):
    print(f"Topik {idx}: {topic}")
# lda_model.show_topics(num_topics=total_topics, num_words=number_words)

# def word_count(documents):
#     word_counts = {}
#     for doc in documents:
#         # words = doc.lower().split()  # Konversi ke huruf kecil dan split kata
#         for word in doc:
#             if word in word_counts:
#                 word_counts[word] += 1
#             else:
#                 word_counts[word] = 1
#     return word_counts

# # Hitung jumlah kata
# counts = word_count(doc_clean)

# # Tampilkan hasil
# for word, count in counts.items():
#     print(f"{word}: {count}")
from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in doc_clean for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])
# print("data\n")
# print(doc_clean)
# df_imp_wcount = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count']) 
# print(df_imp_wcount)

# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# text = " ".join(doc_clean)

# def get_word_cloud(df, c):
#     cm = ' '
#     s_word = set(STOPWORDS)
    
#     for sent in df[c]:
#         # ''' converting sent into string '''
#         sent = str(sent)
#         # ''' spiltting every sent from (" ") '''
#         tokens = sent.split()
        
#         for i in range(len(tokens)):
#             tokens[i] = tokens[i].lower()
        
#         # ''' joining all tokesn '''
#         cm += " ".join(tokens)
    
#     word_cloud = WordCloud(width=800, height=400, background_color='black', stopwords=s_word,
#                            min_font_size=10).generate(cm)
    
#     plt.figure(figsize = (10, 10), facecolor = None) 
#     plt.imshow(word_cloud) 
#     plt.axis("off")
#     plt.tight_layout(pad = 0) 
#     plt.show()

# get_word_cloud(doc_clean,'text')

# Membuat WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Menampilkan WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()