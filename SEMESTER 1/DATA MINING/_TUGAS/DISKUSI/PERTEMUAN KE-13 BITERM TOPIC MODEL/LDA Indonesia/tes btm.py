import pandas as pd
from biterm.utility import vec_to_biterms, topic_summuary
from biterm.btm import oBTM
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import numpy as np
from gensim.models.coherencemodel import CoherenceModel

# Ensure you have the required NLTK data
nltk.download('stopwords')

# Load your data
data = ["sample text data", "more sample text", "and so on..."]

# Initialize stop words
stop_words = set(stopwords.words('english'))

# Preprocess function to remove stop words
def preprocess(text):
    tokens = text.lower().split()
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
processed_data = [preprocess(text) for text in data]

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_data).toarray()
vocab = vectorizer.get_feature_names_out()

# Create biterms
biterms = vec_to_biterms(X)

# Create and fit the BTM model
btm = oBTM(num_topics=4, V=vocab)
btm.fit(biterms, iterations=100)

# Transform the data
topics = btm.transform(biterms)

# Coherence Score
topic_words = []
for topic_dist in btm.phi_wz.T:
    topic_words.append([vocab[i] for i in np.argsort(topic_dist)[:-11:-1]])

cm = CoherenceModel(topics=topic_words, texts=[doc.split() for doc in processed_data], dictionary=vectorizer.vocabulary_, coherence='c_v')
coherence = cm.get_coherence()
print("Coherence Score: ", coherence)

# Average Entropy
def compute_entropy(distribution):
    return -np.sum(distribution * np.log(distribution + 1e-10))

entropy = np.mean([compute_entropy(doc_topic) for doc_topic in topics])
print("Average Entropy: ", entropy)

# Perplexity
def compute_perplexity(model, docs, vocab):
    phi = model.phi_wz
    theta = model.theta_zd
    
    perplexity = 0
    word_count = 0

    for doc in docs:
        doc_word_count = sum(doc)
        word_count += doc_word_count
        for word_id in range(len(vocab)):
            word_prob = np.sum(theta[:, doc] * phi[word_id, :])
            perplexity -= np.log(word_prob + 1e-10) * doc[word_id]

    perplexity = np.exp(perplexity / word_count)
    return perplexity

perplexity = compute_perplexity(btm, X, vocab)
print("Perplexity: ", perplexity)
