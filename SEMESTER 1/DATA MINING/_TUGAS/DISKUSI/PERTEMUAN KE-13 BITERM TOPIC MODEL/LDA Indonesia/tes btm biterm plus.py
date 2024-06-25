import bitermplus as btm
import numpy as np
import pandas as pd
import re,string
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
# IMPORTING DATA
df = pd.read_excel("dataBerita.xlsx")

def remove_tweet_special(text):
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://", " ").replace("https://", " ")
                
df['textdata'] = df['textdata'].apply(remove_tweet_special)

def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['textdata'] = df['textdata'].apply(remove_number)

#remove tanda baca
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

df['textdata'] = df['textdata'].apply(remove_punctuation)

#remove whitespace 
def remove_whitespace_LT(text):
    return text.strip()
df['textdata'] = df['textdata'].apply(remove_whitespace_LT)

def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

df['textdata'] = df['textdata'].apply(remove_singl_char)

# NLTK word tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

df['textdata_tokens'] = df['textdata'].apply(word_tokenize_wrapper)



texts = df["textdata"].str.strip().to_list()

# PREPROCESSING
# Obtaining terms frequency in a sparse matrix and corpus vocabulary
X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
tf = np.array(X.sum(axis=0)).ravel()
# Vectorizing documents
docs_vec = btm.get_vectorized_docs(texts, vocabulary)
docs_lens = list(map(len, docs_vec))
# Generating biterms
biterms = btm.get_biterms(docs_vec)

# INITIALIZING AND RUNNING MODEL
model = btm.BTM(
    X, vocabulary, seed=12321, T=8, M=20, alpha=50/8, beta=0.01)
model.fit(biterms, iterations=20)
p_zd = model.transform(docs_vec)

# METRICS
perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
# or
perplexity = model.perplexity_
coherence = model.coherence_

# LABELS
model.labels_
# or
btm.get_docs_top_topic(texts, model.matrix_docs_topics_)



print("coherence :")
print(coherence)

print(model.labels_)
print("\n")
print(btm.get_docs_top_topic(texts, model.matrix_docs_topics_))

print("perplexity :")
print(perplexity)

# import tomotopy as tp

# # Using tomotopy for graphical representation
# corpus = [text.split() for text in texts]
# mdl = tp.LDAModel(k=8, alpha=0.1, eta=0.01)
# for doc in corpus:
#     mdl.add_doc(doc)

# mdl.train(0)
# print('Num of docs:', len(mdl.docs))
# print('Vocab size:', len(mdl.used_vocabs))
# print('Num of words:', mdl.num_words)

# for i in range(100):
#     mdl.train(10)
#     print('Iteration:', i, 'LL:', mdl.ll_per_word)

# # Visualizing the results
# import pyLDAvis
# import pyLDAvis.tomotopy as tomotopyvis

# vis_data = tomotopyvis.prepare(mdl, mdl.corpus)
# pyLDAvis.show(vis_data)

# # Top words in topics
# for k in range(mdl.k):
#     print('Topic #{}'.format(k), mdl.get_topic_words(k, top_n=10))
# # mdl = tp.LDAModel(k=20)
# # for line in open(texts["textdata"]):
# #     mdl.add_doc(line.strip().split())

# # for i in range(0, 100, 10):
# #     mdl.train(10)
# #     print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

# # for k in range(mdl.k):
# #     print('Top 10 words of topic #{}'.format(k))
# #     print(mdl.get_topic_words(k, top_n=10))

# # mdl.summary()